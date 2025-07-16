# Employee Salary Prediction

This project is a web application that predicts whether an employee's annual income exceeds $50,000 based on demographic and employment data. The prediction is made using a machine learning model trained on the Adult Census Income dataset.

## Features

*   Predicts income level (`>50K` or `<=50K`).
*   Provides a confidence score for the prediction.
*   Interactive web interface built with Streamlit.
*   Displays a preview of the dataset.

## Project Structure

```
.
├── adult_3.csv                   # The dataset used for training the model.
├── app.py                        # The main Streamlit application file.
├── main.py                       # Contains the SalaryPredictor class for model training and prediction.
├── requirements.txt              # Lists the Python dependencies for the project.
├── Employee_Salary_Prediction.ipynb # Jupyter notebook for exploration (optional).
└── README.md                     # This file.
```

## Setup and Installation

Follow these steps to set up and run the project on your local machine.

### Prerequisites

*   Python 3.8 or higher
*   pip (Python package installer)

### Installation

1.  **Download the source code:**
    Download the project files and place them in a directory on your computer.

2.  **Navigate to the project directory:**
    Open a terminal or command prompt and navigate to the directory where you saved the project files.
    ```bash
    cd path\to\EMP_Salary_pridiction
    ```

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Application

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your default web browser.

## Application Output

The application will open in your web browser and display the following:

*   **Title:** "Employee Salary Prediction"
*   **Model Accuracy:** A statement showing the accuracy of the trained model (e.g., "The model has an accuracy of 85.51%").
*   **Sidebar for Input:** On the left side, a sidebar titled "Enter Employee Information" will have several input fields:
    *   Sliders for `Age`, `Education Level`, and `Hours per Week`.
    *   Dropdown menus for `Work Class`, `Marital Status`, `Occupation`, etc.
    *   Number input fields for `Capital Gain` and `Capital Loss`.
*   **Prediction Result:** In the main area, a section titled "Prediction Result" will show:
    *   The predicted income (e.g., `Predicted Income: >50K`).
    *   The confidence percentage of the prediction.
*   **Data Preview:** A section titled "Data Preview" will show the first few rows of the dataset in a table.

*   <img width="1365" height="767" alt="image" src="https://github.com/user-attachments/assets/370e8e08-b955-4768-bd75-55af58c5cba3" />


