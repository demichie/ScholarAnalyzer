import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from collections import Counter, defaultdict
from pyvis.network import Network
import streamlit.components.v1 as components
from datetime import datetime
import time
import random
import os
import json
from difflib import SequenceMatcher
from itertools import combinations
from scholarly import scholarly
from wordcloud import WordCloud, STOPWORDS
import re

# --- CONFIGURATION & SETUP ---
DATA_DIR = "data"
STALE_DATA_THRESHOLD_DAYS = 30
DEFAULT_SIMILARITY_THRESHOLD = 0.75
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(
    page_title="ScholarAnalyzer",
    page_icon="🎓",
    layout="wide"
)


# --- DATA HANDLING & ANALYSIS HELPER FUNCTIONS ---

def scrape_author_data(scholar_id):
    """
    Performs the actual scraping of an author's profile from Google Scholar.
    Args: scholar_id (str). Returns: dict | None.
    """
    try:
        author = scholarly.search_author_id(scholar_id)
        filled_author = scholarly.fill(author, sections=['basics', 'indices', 'counts', 'publications'])
        publications = filled_author['publications']
        total_pubs = len(publications)
        filled_pubs = []
        status_text = st.empty()
        progress_bar = st.progress(0)
        for i, pub in enumerate(publications):
            status_text.text(f"Fetching publication {i+1} of {total_pubs}...")
            filled_pub = scholarly.fill(pub)
            filled_pubs.append(filled_pub)
            progress_bar.progress((i + 1) / total_pubs)
            time.sleep(random.uniform(0.5, 1.5))
        status_text.success(f"Successfully fetched all {total_pubs} publications!")
        time.sleep(2)
        status_text.empty()
        progress_bar.empty()
        filled_author['publications'] = filled_pubs
        return filled_author
    except Exception as e:
        st.error(f"Could not retrieve data. Error: {e}")
        return None

def load_data(scholar_id, force_refresh=False):
    """
    Loads author data, prioritizing local files over scraping.
    Args: scholar_id (str), force_refresh (bool). Returns: dict | None.
    """
    file_path = os.path.join(DATA_DIR, f"{scholar_id}.json")
    if os.path.exists(file_path) and not force_refresh:
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
        if 'coauthor_map' not in saved_data: saved_data['coauthor_map'] = {}
        return saved_data
    author_data = scrape_author_data(scholar_id)
    if author_data:
        data_to_save = {"metadata": {"last_updated": datetime.now().isoformat()}, "author_data": author_data, "coauthor_map": {}}
        save_data(scholar_id, data_to_save)
        return data_to_save
    return None

def save_data(scholar_id, data):
    """
    Saves the complete data structure to a local JSON file.
    Args: scholar_id (str), data (dict). Returns: None.
    """
    file_path = os.path.join(DATA_DIR, f"{scholar_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)

def get_acuna_model_inputs(author_data):
    """
    Extracts parameters for the Acuna et al. (2012) prediction model.
    Args: author_data (dict). Returns: dict.
    """
    top_journals = ['nature', 'science', 'pnas', 'proceedings of the national academy of sciences', 'neuron']
    h = author_data.get('hindex', 0)
    n = len(author_data['publications'])
    pub_years, venues, q = [], set(), 0
    for pub in author_data['publications']:
        bib = pub.get('bib', {})
        year_value = bib.get('pub_year')
        if year_value:
            try: pub_years.append(int(year_value))
            except (ValueError, TypeError): pass
        
        pub_source = (bib.get('journal') or bib.get('venue') or "").lower()
        if pub_source:
            venues.add(pub_source)
            if any(top_journal in pub_source for top_journal in top_journals):
                q += 1
    
    first_pub_year = min(pub_years) if pub_years else datetime.now().year
    y = datetime.now().year - first_pub_year
    j = len(venues)
    return {'h': h, 'n': n, 'y': y, 'j': j, 'q': q}

def get_raw_coauthors(author_data):
    """
    Extracts a raw list of all co-authors from the publication data.
    Args: author_data (dict). Returns: list.
    """
    all_coauthors_raw = []
    main_author_name = author_data['name']
    for pub in author_data['publications']:
        authors = pub['bib'].get('author')
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.replace(' and ', ',').split(',')]
        if authors and isinstance(authors, list):
            for author in authors:
                if author and author.lower() != main_author_name.lower():
                    all_coauthors_raw.append(author)
    return all_coauthors_raw

def apply_automatic_merges(raw_coauthors, coauthor_map, similarity_threshold):
    """
    Automatically identifies and merges similar co-author names.
    Args: raw_coauthors (list), coauthor_map (dict), similarity_threshold (float). Returns: dict.
    """
    raw_counts = Counter(raw_coauthors)
    unique_raw_names = list(raw_counts.keys())
    similar_pairs = find_similar_names(unique_raw_names, similarity_threshold)
    for name1, name2 in similar_pairs:
        canonical_name = name1 if raw_counts[name1] >= raw_counts[name2] else name2
        variant_name = name2 if canonical_name == name1 else name1
        if variant_name not in coauthor_map:
            coauthor_map[variant_name] = canonical_name
    return coauthor_map

def find_similar_names(names_list, threshold):
    """
    Identifies pairs of names that are likely duplicates based on string similarity.
    Args: names_list (list), threshold (float). Returns: list.
    """
    similar_pairs = []
    for name1, name2 in combinations(names_list, 2):
        if SequenceMatcher(None, name1, name2).ratio() > threshold:
            similar_pairs.append(tuple(sorted((name1, name2))))
    return sorted(list(set(similar_pairs)))

def create_coauthor_graph(author_name, coauthor_counts, min_pubs):
    """
    Creates an interactive co-author network graph using pyvis.
    Args: author_name (str), coauthor_counts (Counter), min_pubs (int). Returns: str | None.
    """
    filtered_coauthors = {name: count for name, count in coauthor_counts.items() if count >= min_pubs}
    if not filtered_coauthors: return None
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, cdn_resources='in_line')
    net.add_node(0, label=author_name, color="#ff4b4b", size=30, title=f"Profiled Author: {author_name}")
    for i, (name, count) in enumerate(filtered_coauthors.items()):
        node_id = i + 1
        node_size = 10 + 5 * np.log1p(count)
        net.add_node(node_id, label=name, size=node_size, title=f"{count} shared publications")
        net.add_edge(0, node_id, value=count, title=f"{count} collaborations")
    graph_file = "coauthor_graph.html"
    try:
        net.save_graph(graph_file)
        return graph_file
    except Exception as e:
        st.error(f"Error creating graph: {e}")
        return None

def generate_wordcloud(text, max_words, colormap, custom_stopwords):
    """
    Generates and returns a word cloud image from a given text.
    Args: text (str), max_words (int), colormap (str), custom_stopwords (list). Returns: A PIL Image object.
    """
    stopwords = set(STOPWORDS)
    stopwords.update(custom_stopwords)

    wordcloud = WordCloud(
        width=1600, height=800, background_color='white', stopwords=stopwords,
        max_words=max_words, colormap=colormap, contour_color='steelblue', contour_width=2
    ).generate(text)
    return wordcloud.to_image()


# --- STREAMLIT USER INTERFACE ---

st.title("🎓 ScholarAnalyzer")
st.markdown("An interactive dashboard to analyze, visualize, and forecast Google Scholar profiles.")
with st.expander("❓ How to find a Google Scholar ID?"):
    st.write("""
    1. Go to the Google Scholar profile page of the person you are interested in.
    2. Look at the URL in your browser's address bar.
    3. The URL will have a structure similar to `https://scholar.google.com/citations?user=XXXXXXXXXXXX&hl=en`
    4. The profile ID is the string of alphanumeric characters after `user=`.
    
    **Example:** For the URL `https://scholar.google.com/citations?user=qc6CJjYAAAAJ`, the ID is `qc6CJjYAAAAJ`.
    """)

scholar_id_input = st.text_input("Enter Google Scholar ID:", placeholder="E.g., qc6CJjYAAAAJ")
force_refresh = st.checkbox("Force refresh from Google Scholar (ignore local data)")

if st.button("Analyze Profile"):
    if scholar_id_input:
        if 'scholar_id' in st.session_state and st.session_state.scholar_id != scholar_id_input:
            for key in list(st.session_state.keys()): del st.session_state[key]
        with st.spinner("Loading data..."):
            full_data = load_data(scholar_id_input, force_refresh)
            if full_data:
                st.session_state.full_data = full_data
                st.session_state.scholar_id = scholar_id_input
                st.session_state.coauthor_map = full_data.get('coauthor_map', {})
            else:
                st.error("Failed to load data.")
    else:
        st.warning("Please enter a Google Scholar ID.")

if 'full_data' in st.session_state:
    full_data = st.session_state.full_data
    author_data = full_data['author_data']
    scholar_id = st.session_state.scholar_id
    main_author_name = author_data['name']
    
    has_venue_data = any('journal' in pub.get('bib', {}) or 'venue' in pub.get('bib', {}) for pub in author_data.get('publications', []))
    if not has_venue_data and author_data.get('publications'):
        st.warning(
            "**Warning:** The loaded data file appears to be from an older version and is missing "
            "detailed journal information. Features like 'Distinct Journals' and "
            "'Top Journals' may not work correctly. Please use the 'Force refresh' option "
            "to download the complete data.",
            icon="⚠️"
        )

    tabs = st.tabs(["📊 Profile", "📚 Publications", "📈 H-Index Forecast", "⏱️ Temporal Analysis", "👥 Co-Authors", "☁️ Word Cloud"])
    
    with tabs[0]: # Profile Tab
        st.header(f"Profile of {main_author_name}")
        col1, col2 = st.columns([1, 4])
        try:
            with col1: st.image(author_data['url_picture'], width=150)
        except:
            with col1: st.image("./scholarImage.png", width=150)
        with col2:
            st.write(f"**Affiliation:** {author_data.get('affiliation', 'N/A')}")
            st.write(f"**Interests:** {', '.join(author_data.get('interests', []))}")
            st.markdown(f"[Link to Google Scholar Profile](https://scholar.google.com/citations?user={scholar_id})")
        st.divider()
        st.subheader("Key Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Citations", author_data.get('citedby', 0))
        m2.metric("h-index", author_data.get('hindex', 0))
        m3.metric("i10-index", author_data.get('i10index', 0))
        st.subheader("Yearly Citation Trends")
        if 'cites_per_year' in author_data and author_data['cites_per_year']:
            cites_df = pd.DataFrame(list(author_data['cites_per_year'].items()), columns=['Year', 'Citations'])
            cites_df = cites_df.sort_values('Year')
            fig = px.bar(cites_df, x='Year', y='Citations', title="Citations Received per Year")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]: # Publications Tab
        st.header("List of Publications")
        if 'publications' in author_data and author_data['publications']:
            pubs_list = []
            for pub in author_data['publications']:
                bib = pub.get('bib', {})
                author_info = bib.get('author')
                authors_str = ', '.join(author_info) if isinstance(author_info, list) else author_info or 'N/A'
                venue_str = bib.get('journal') or bib.get('venue') or 'N/A'
                pubs_list.append({'Title': bib.get('title', 'N/A'), 'Authors': authors_str, 'Year': bib.get('pub_year'), 'Citations': pub.get('num_citations', 0), 'Journal': venue_str})
            pubs_df = pd.DataFrame(pubs_list)
            pubs_df['Year'] = pd.to_numeric(pubs_df['Year'], errors='coerce').fillna(0).astype(int)
            st.dataframe(pubs_df, use_container_width=True, height=600)

    with tabs[2]: # H-Index Forecast Tab
        st.header("Future H-Index Forecast")
        st.info(
            """
            **Disclaimer and Model Interpretation**

            This forecast is based on the statistical model published by **Acuna, D. E., Allesina, S., & Kording, K. P. (2012) in *Nature***.
            
            **Important Considerations:**
            - **Decreasing Accuracy:** The model's accuracy decreases significantly over longer time horizons. The authors report R² values of 0.92 (1 year), 0.67 (5 years), and just 0.48 (10 years). The 10-year forecast should be considered highly speculative.
            - **The 10-Year Paradox:** You may notice that the 10-year prediction is sometimes lower than the 5-year one. This is not an error. It's a characteristic of the model, which heavily penalizes career length (`y`) and gives less weight to the current h-index (`h`) in its long-term forecast. It suggests a potential "plateau" in growth but does not mean the actual h-index will decrease.
            - **Context:** The model was primarily trained on life scientists and may be less accurate for other fields.
            """
        )
        params = get_acuna_model_inputs(author_data)
        h, n, y, j, q = params['h'], params['n'], params['y'], params['j'], params['q']
        st.subheader("Current Model Parameters")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Current h-index (h)", h); c2.metric("# Articles (n)", n); c3.metric("Years since first pub (y)", y); c4.metric("# Distinct Journals (j)", j); c5.metric("# Top Journal Articles (q)", q)
        st.divider()
        if n > 0:
            h1, h5, h10 = (0.76+(0.37*np.sqrt(n))+(0.97*h)-(0.07*y)+(0.02*j)+(0.03*q)), (4.00+(1.58*np.sqrt(n))+(0.86*h)-(0.35*y)+(0.06*j)+(0.20*q)), (8.73+(1.33*np.sqrt(n))+(0.48*h)-(0.41*y)+(0.52*j)+(0.82*q))
            st.subheader("H-Index Forecast Metrics")
            f1, f2, f3 = st.columns(3)
            f1.metric("Predicted h-index in 1 Year", f"{h1:.1f}", f"{h1-h:.1f} (R²=0.92)"); f2.metric("Predicted h-index in 5 Years", f"{h5:.1f}", f"{h5-h:.1f} (R²=0.67)"); f3.metric("Predicted h-index in 10 Years", f"{h10:.1f}", f"{h10-h:.1f} (R²=0.48)")
            st.subheader("H-Index Growth Visualization")
            forecast_data = {'Timeframe': ['Current', 'In 1 Year', 'In 5 Years', 'In 10 Years'], 'H-Index': [h, h1, h5, h10], 'Type': ['History', 'Forecast', 'Forecast', 'Forecast']}
            forecast_df = pd.DataFrame(forecast_data).assign(H_Index=lambda df: df['H-Index'].clip(lower=0))
            fig_forecast = px.bar(forecast_df, x='Timeframe', y='H-Index', color='Type', title="H-Index Growth Forecast", text_auto='.1f')
            fig_forecast.update_traces(textposition='outside'); st.plotly_chart(fig_forecast, use_container_width=True)

    with tabs[3]: # Temporal Analysis Tab
        st.header("Temporal Citation Analysis")
        st.markdown("Analyze how many citations each article received within a specific time frame.")
        if 'cites_per_year' in author_data and author_data['cites_per_year']:
            valid_years = {int(y) for pub in author_data['publications'] for y in pub.get('cites_per_year', {}).keys()}
            if valid_years:
                years = sorted(list(valid_years))
                selected_start, selected_end = st.select_slider("Select year range:", options=years, value=(years[0], years[-1]))
                start_year, end_year = min(selected_start, selected_end), max(selected_start, selected_end)
                st.write(f"Analyzing citations between **{start_year}** and **{end_year}**.")
                results = [{'Title': pub['bib'].get('title', 'N/A'), 'Citations in Range': sum(c for y, c in pub.get('cites_per_year', {}).items() if start_year <= int(y) <= end_year), 'Total Citations': pub.get('num_citations', 0)} for pub in author_data['publications']]
                results_df = pd.DataFrame([r for r in results if r['Citations in Range'] > 0]).sort_values('Citations in Range', ascending=False)
                if not results_df.empty: st.dataframe(results_df.reset_index(drop=True), use_container_width=True)
                else: st.info(f"No articles received citations between {start_year} and {end_year}.")

    with tabs[4]:
        st.header(f"Co-Author Analysis for {main_author_name}")
        
        raw_coauthors = get_raw_coauthors(author_data)
        
        with st.expander("🛠️ Co-Author Management & Cleaning"):
            similarity_threshold = st.slider("Similarity threshold for auto-merging:", 0.50, 1.0, DEFAULT_SIMILARITY_THRESHOLD, 0.05, help="Lower values will merge more names automatically.")
            if st.button("Apply Auto-Merges & Refresh"):
                st.session_state.coauthor_map = apply_automatic_merges(raw_coauthors, st.session_state.coauthor_map, similarity_threshold)
                st.rerun()

            st.subheader("Current Merge Rules")
            reversed_map = defaultdict(list)
            for variant, canonical in st.session_state.coauthor_map.items():
                reversed_map[canonical].append(variant)
            if not reversed_map:
                st.write("No merge rules are currently active.")
            else:
                for canonical, variants in reversed_map.items():
                    display_name = f"{canonical} (Analyzed Scholar)" if canonical == main_author_name else canonical
                    st.write(f"**{display_name}** receives merges from:")
                    for variant in variants:
                        col1, col2 = st.columns([4, 1])
                        col1.markdown(f"- `{variant}`")
                        if col2.button("Unmerge", key=f"unmerge_{variant}", use_container_width=True):
                            del st.session_state.coauthor_map[variant]; st.rerun()
            
            st.subheader("Manually Merge Co-Authors")
            # The list for manual merging should always contain all unique raw names, regardless of filters
            normalized_for_merge = [st.session_state.coauthor_map.get(name, name) for name in raw_coauthors if st.session_state.coauthor_map.get(name, name) != main_author_name]
            active_author_names = sorted(list(set(normalized_for_merge)))
            
            col1, col2 = st.columns(2)
            names_to_merge = col1.multiselect("Merge these names...", options=active_author_names, key="merge_from")
            
            canonical_author_label = f"{main_author_name} (Analyzed Scholar)"
            canonical_name_options = [canonical_author_label] + [name for name in active_author_names if name not in names_to_merge]
            canonical_name_selection = col2.selectbox("...into this name.", options=canonical_name_options, index=None, key="merge_to")
            
            if st.button("Apply Manual Merge"):
                if names_to_merge and canonical_name_selection:
                    canonical_name = main_author_name if canonical_name_selection == canonical_author_label else canonical_name_selection
                    for name_to_merge in names_to_merge:
                        for raw_name in raw_coauthors:
                            if st.session_state.coauthor_map.get(raw_name, raw_name) == name_to_merge:
                                st.session_state.coauthor_map[raw_name] = canonical_name
                        st.session_state.coauthor_map[name_to_merge] = canonical_name
                    st.rerun()

            if st.button("Save Co-Author Changes", type="primary"):
                full_data['coauthor_map'] = st.session_state.coauthor_map
                save_data(scholar_id, full_data)
                st.success("Co-author map saved to JSON file!")

        st.divider()
        st.subheader("Filter and Display Collaborators")

        # --- NEW: Slider for minimum citations ---
        max_citations = 0
        if author_data['publications']:
            max_citations = max(pub.get('num_citations', 0) for pub in author_data['publications'])
        
        min_citations_filter = st.slider(
            "Only consider publications with at least N citations:",
            min_value=0,
            max_value=int(max_citations),
            value=0,
            help="Filter collaborations based on the impact of the shared papers."
        )

        # Filter publications based on the citation slider
        filtered_pubs = [
            pub for pub in author_data['publications'] 
            if pub.get('num_citations', 0) >= min_citations_filter
        ]
        
        # Re-calculate raw co-authors from the filtered publication list
        filtered_raw_coauthors = get_raw_coauthors({'name': main_author_name, 'publications': filtered_pubs})
        
        # Apply the existing merge map to the filtered co-author list
        normalized_coauthors = [
            st.session_state.coauthor_map.get(name, name) for name in filtered_raw_coauthors 
            if st.session_state.coauthor_map.get(name, name) != main_author_name
        ]
        coauthor_counts = Counter(normalized_coauthors)
        coauthor_df = pd.DataFrame(coauthor_counts.items(), columns=['Co-Author', 'Publication Count']).sort_values('Publication Count', ascending=False)
        
        st.subheader("Collaborator List (Filtered)")
        st.write(f"Displaying collaborators from {len(filtered_pubs)} publications with at least {min_citations_filter} citations.")
        st.dataframe(coauthor_df.reset_index(drop=True), use_container_width=True)
        
        st.divider()
        st.subheader("Co-Author Network")

        if not coauthor_df.empty:
            # The sliders for the graph now operate on the filtered data
            if len(coauthor_df) > 20:
                default_min_pubs = coauthor_df['Publication Count'].iloc[19]
            else:
                default_min_pubs = 1
            
            max_collabs = coauthor_df['Publication Count'].max()
            min_pubs = st.slider(
                "Show co-authors with at least N publications:", 
                min_value=1, 
                max_value=int(max_collabs), 
                value=int(default_min_pubs)
            )
            
            graph_file = create_coauthor_graph(main_author_name, coauthor_counts, min_pubs)
            if graph_file:
                with open(graph_file, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=800)
        else:
            st.info("No co-authors match the current filter criteria.")

    with tabs[5]: # Word Cloud Tab
        st.header("Publication Title Word Cloud")
        st.markdown("This cloud visualizes the most frequent words in publication titles, giving an overview of research topics.")

        st.subheader("Cloud Options")
        
        col1, col2 = st.columns(2)
        with col1:
            max_words = st.slider("Maximum words in cloud:", 50, 500, 200, 10)
            colormap = st.selectbox("Color scheme:", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

        with col2:
            pub_years_set = set()
            for pub in author_data['publications']:
                year_value = pub.get('bib', {}).get('pub_year')
                if year_value:
                    try:
                        pub_years_set.add(int(year_value))
                    except (ValueError, TypeError):
                        pass
            pub_years = sorted(list(pub_years_set))
            
            if pub_years:
                start_year, end_year = st.select_slider(
                    "Filter publications by year:",
                    options=pub_years,
                    value=(pub_years[0], pub_years[-1])
                )
            else:
                start_year, end_year = None, None
                st.info("No publication years available for filtering.")

        custom_stopwords_input = st.text_area("Add custom words to ignore (comma-separated):", "et, al, review, based, study, analysis")
        
        if start_year and end_year:
            st.write(f"Displaying word cloud for publications from **{start_year}** to **{end_year}**.")
            filtered_pubs = []
            for pub in author_data['publications']:
                try:
                    pub_year = int(pub.get('bib', {}).get('pub_year', ''))
                    if start_year <= pub_year <= end_year:
                        filtered_pubs.append(pub)
                except (ValueError, TypeError):
                    continue
        else:
            filtered_pubs = author_data['publications']

        custom_stopwords = [word.strip().lower() for word in custom_stopwords_input.split(',')]
        
        titles = [pub['bib'].get('title', '') for pub in filtered_pubs]
        full_text = ' '.join(titles)

        if full_text.strip():
            clean_text = re.sub(r'[^\w\s]', '', full_text).lower()
            
            with st.spinner("Generating word cloud..."):
                wordcloud_image = generate_wordcloud(clean_text, max_words, colormap, custom_stopwords)
                st.image(wordcloud_image, use_container_width=True, caption=f"Word Cloud for Titles ({start_year}-{end_year})" if start_year else "Word Cloud of All Publication Titles")
        else:
            st.info("No publication titles found in the selected year range to generate a word cloud.")
