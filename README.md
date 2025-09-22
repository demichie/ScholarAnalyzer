# ScholarAnalyzer: Google Scholar Profile Analyzer

*Developed by: [Il Tuo Nome]*
*Core code structure and functionality generated with assistance from Google AI Studio.*

---

> A powerful Streamlit dashboard to analyze, visualize, and forecast the academic impact of a Google Scholar profile.

ScholarAnalyzer is an interactive web application that fetches data directly from a Google Scholar profile and provides in-depth insights into a researcher's publication history, citation trends, and collaborative network. It features advanced tools for data cleaning and forecasting, making it a comprehensive solution for academic analysis.

## Key Features

-   **Comprehensive Profile Dashboard**: At-a-glance view of key metrics like h-index, i10-index, and total citations, with a historical graph of citation trends.
-   **Detailed Publication List**: A filterable table of all publications.
-   **H-Index Forecasting**: Predicts the future evolution of an author's h-index using a model published in *Nature*, complete with a detailed disclaimer and visual chart.
-   **Temporal Citation Analysis**: Discover which papers had the most impact within a specific date range.
-   **Advanced Co-Author Analysis**:
    -   **Smart Merging**: Automatically identifies and merges similar co-author names with a user-configurable similarity threshold.
    -   **Full Manual Control**: An intuitive interface to manually merge multiple names, un-merge incorrect associations, and even merge variants into the main scholar's name.
    -   **Intelligent Network Graph**: Visualizes the collaborative network with dual filters (publication count and citation impact) and an intelligent default view to prevent clutter.
-   **Dynamic Word Cloud**:
    -   Visualize the key topics from publication titles.
    -   Interactively filter by year range to see how research focus has evolved over time.
    -   Customize the appearance with options for word count, color scheme, and custom stopwords.
-   **Persistent Data Caching**: Fetched profile data and all co-author merging rules are saved locally to ensure fast re-loading and to minimize requests to Google Scholar.

## Screenshots

*(It is highly recommended to add screenshots of your application here to showcase its features)*

| Co-Author Management Interface                      | Co-Author Network Graph                                 |
| --------------------------------------------------- | ------------------------------------------------------- |
| ![Co-Author Management](path/to/management_image.png) | ![Co-Author Graph](path/to/graph_image.png)             |
| **Word Cloud with Filters**                         | **H-Index Forecast**                                    |
| ![Word Cloud](path/to/wordcloud_image.png)          | ![H-Index Forecast Chart](path/to/forecast_image.png) |

## Setup and Installation

This project is managed using Conda for environment reproducibility.

1.  **Prerequisites**:
    -   Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/[your-username]/ScholarAnalyzer.git
    cd ScholarAnalyzer
    ```

3.  **Create the Conda Environment**:
    The `environment.yml` file contains all the necessary dependencies. Create and activate the environment with the following commands:
    ```bash
    # Create the environment from the file
    conda env create -f environment.yml

    # Activate the new environment
    conda activate scholar_dashboard
    ```

## How to Run

With the `scholar_dashboard` environment activated, run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

The application should automatically open in your web browser.

## Core Functionality

The dashboard is organized into six powerful tabs:

1.  **Profile**: Displays general statistics and yearly citation trends.
2.  **Publications**: A complete list of the author's publications.
3.  **H-Index Forecast**: Provides 1, 5, and 10-year forecasts for the h-index, including a detailed disclaimer about the model's limitations.
4.  **Temporal Analysis**: Allows you to see which papers were most cited within a user-defined time window.
5.  **Co-Authors**: Features a comprehensive suite for analyzing and cleaning co-author data, with automatic and manual merging tools and a dual-filtered network graph.
6.  **Word Cloud**: Generates a dynamic word cloud from publication titles, filterable by year to track the evolution of research topics.

## Attributions and References

### Author

-   **[Il Tuo Nome]**

### AI-Assisted Development

The Python code for this project was generated with significant assistance from **Google AI Studio**.

### Bibliographic Reference

The h-index forecast feature is based on the predictive model described in the following publication:

> Acuna, D. E., Allesina, S., & Kording, K. P. (2012). Predicting scientific success. *Nature*, 489(7415), 201–202. [https://doi.org/10.1038/489201a](https://doi.org/10.1038/489201a)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```
