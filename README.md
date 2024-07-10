

# Book Recommendation Agent

This project is a Book Recommendation Agent that leverages Streamlit and BERT-based NLP techniques to recommend books based on user input. Users can specify a genre and receive tailored book recommendations based on data from a Goodreads dataset.

## Features

- **Data Loading**: Efficiently loads and displays data from a CSV file.
- **Top 100 Books Selection**: Filters and ranks the top 100 books in the specified genre based on average ratings.
- **Top 10 Books Selection**: Further refines the list to the top 10 books from the top 100.
- **BERT-based Recommendation**: Utilizes the SentenceTransformer model to recommend the best book from the top 10 using cosine similarity.
- **User Interaction**: Provides an interactive and user-friendly interface via Streamlit, enabling users to enter genres and receive real-time recommendations.
- **Deployment**: The application is deployed using ngrok for easy access and testing without complex setup.

## Repository Contents

- `app.py`: Main application file containing the Streamlit app and recommendation logic.
- `goodreads_data.csv`: Dataset containing book information from Goodreads.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file, providing an overview of the project.
- `Book_Recommendation_System.pdf`: Documentation explaining why this model and approach were used.
- `FinalBookRecc.ipynb`: Jupyter notebook with the code used to create the `app.py` file.
- `Demo Video.mp4`: A demo video showing the functionality of the application.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/book-recommendation-agent.git
   cd book-recommendation-agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the `goodreads_data.csv` file in the same directory as `app.py`.

## Usage

1. Set your ngrok authentication token:
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token('YOUR_NGROK_AUTH_TOKEN')
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Once the app is running, ngrok will provide a public URL. Use this URL to access the Streamlit app.

## Example

- **Input**: Enter a genre such as "Science Fiction" or "Fantasy".
- **Output**: The app displays the top 100 books in the specified genre, then narrows it down to the top 10, and finally recommends the best book using BERT-based similarity.

## Contributing

Feel free to contribute to this project by submitting a pull request or opening an issue on GitHub.

## Acknowledgments

- Streamlit
- Sentence Transformers
- Goodreads
- ngrok

Make sure to replace the placeholder `YOUR_NGROK_AUTH_TOKEN` with your actual ngrok authentication token and update the repository URL and any other project-specific details.
# Book-Recommendation-System
