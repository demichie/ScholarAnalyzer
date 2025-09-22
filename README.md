# ScholarAnalyzer: Google Scholar Profile Analyzer

*Developed by: Mattia de' Michieli Vitturi*
*Core code structure and functionality generated with assistance from Google AI Studio.*

---

> A powerful Streamlit dashboard to analyze, visualize, and forecast the academic impact of a Google Scholar profile.

ScholarScope is an interactive web application that fetches data directly from a Google Scholar profile and provides in-depth insights into a researcher's publication history, citation trends, and collaborative network. It features advanced tools for data cleaning and forecasting, making it a comprehensive solution for academic analysis.

## Key Features

-   **Comprehensive Profile Dashboard**: At-a-glance view of key metrics like h-index, i10-index, and total citations, with a historical graph of citation trends.
-   **Detailed Publication List**: A searchable and filterable table of all publications.
-   **H-Index Forecasting**: Predicts the future evolution of an author's h-index using a model published in *Nature*.
-   **Temporal Citation Analysis**: Discover which papers had the most impact within a specific date range.
-   **Advanced Co-Author Analysis**:
    -   Automatically identifies and merges similar co-author names (e.g., "J. Smith" and "John Smith").
    -   Provides an interactive management interface to manually merge, un-merge, and save cleaned co-author data.
    -   Visualizes the collaborative network with an interactive graph, highlighting the most frequent collaborators.
-   **Local Data Caching**: Fetched data is saved locally to ensure fast re-loading and to minimize requests to Google Scholar.

## Screenshots

*(It is highly recommended to add screenshots of your application here)*

| Profile Dashboard                                | Co-Author Network Graph                        |
| ------------------------------------------------ | ---------------------------------------------- |
| ![Profile Dashboard](path/to/profile_image.png)  | ![Co-Author Graph](path/to/graph_image.png)    |
| **Temporal Analysis**                            | **Co-Author Management**                       |
| ![Temporal Analysis](path/to/temporal_image.png) | ![Co-Author Merge](path/to/management_image.png) |

## Setup and Installation

This project is managed using Conda for environment reproducibility.

1.  **Prerequisites**:
    -   Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/[your-username]/ScholarScope.git
    cd ScholarScope
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

The dashboard is organized into several tabs:

1.  **Profile**: Displays general statistics and yearly citation trends.
2.  **Publications**: A complete list of the author's publications.
3.  **H-Index Forecast**: Provides 1, 5, and 10-year forecasts for the h-index based on the model by Acuna et al.
4.  **Temporal Analysis**: Allows you to see which papers were most cited within a user-defined time window.
5.  **Co-Authors**: Features a powerful suite for analyzing and cleaning co-author data, complete with an interactive network graph. Merging rules are saved locally for each profile.

## Attributions and References

### Author

-   **[Il Tuo Nome]**

### AI-Assisted Development

The Python code for this project, including its structure, features, and documentation, was generated with significant assistance from **Google AI Studio**.

### Bibliographic Reference

The h-index forecast feature is based on the predictive model described in the following publication:

> Acuna, D. E., Allesina, S., & Kording, K. P. (2012). Predicting scientific success. *Nature*, 489(7415), 201–202. [https://doi.org/10.1038/489201a](https://doi.org/10.1038/489201a)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
```
