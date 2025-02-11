import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def read_prediction_smiles(file_path='Property_based_Model/torch_code/'+'prediction_smiles.csv'):
    # Print current working directory
    print("Current working directory:", os.getcwd())
    
    # Print absolute path of the file
    abs_file_path = os.path.abspath(file_path)
    print("Attempting to read file:", abs_file_path)
    
    # Check if the file exists
    if not os.path.exists(abs_file_path):
        raise FileNotFoundError(f"The file {abs_file_path} does not exist.")
    
    # Read the CSV file
    df = pd.read_csv(abs_file_path)
    
    # Sort the dataframe by 'Er' column in descending order and keep the first occurrence of each SMILES
    df_unique = df.sort_values('Er', ascending=False).drop_duplicates(subset='Smiles', keep='first')
    
    # Reset the index of the resulting dataframe
    df_unique = df_unique.reset_index(drop=True)
    
    # Select only 'Tg' and 'Er' columns
    df_unique = df_unique[['Smiles', 'Tg', 'Er']]
    
    # Display the first few rows of the unique dataframe
    print(df_unique.head())
    
    # Display basic information about the unique dataframe
    print(df_unique.info())
    print(df_unique.shape)
    
    return df_unique

def draw_parallel_coordinates(df):
    # Create the parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line = dict(color = df['Er'],
                        colorscale = 'Viridis',
                        showscale = True),
            dimensions = list([
                dict(range = [df['Tg'].min(), df['Tg'].max()],
                     label = 'Tg', values = df['Tg']),
                dict(range = [df['Er'].min(), df['Er'].max()],
                     label = 'Er', values = df['Er'])
            ])
        )
    )

    # Update the layout
    fig.update_layout(
        title = 'Parallel Coordinates Plot of Tg and Er',
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
    )

    # Show the plot
    fig.show()

def create_heatmap(df):
    # Select only 'Tg' and 'Er' columns for the heatmap
    columns_for_heatmap = ['Tg', 'Er']
    
    # Create the heatmap
    fig = px.imshow(df[columns_for_heatmap].corr(),
                    x=columns_for_heatmap,
                    y=columns_for_heatmap,
                    color_continuous_scale='Viridis',
                    title='Correlation Heatmap of Tg and Er')
    
    # Update the layout
    fig.update_layout(
        width=600,
        height=600,
        xaxis_title='Features',
        yaxis_title='Features'
    )
    
    # Show the plot
    fig.show()

def create_scatter_plot(df):
    # Create the scatter plot
    fig = px.scatter(df, x='Tg', y='Er', hover_data=['Smiles'],
                     title='Scatter Plot of Tg vs Er',
                     labels={'Tg': 'Glass Transition Temperature (Tg)',
                             'Er': 'Dielectric Constant (Er)'},
                     color='Er',  # Color points based on Er value
                     color_continuous_scale='Viridis')

    # Update the layout
    fig.update_layout(
        width=800,
        height=600,
        xaxis_title='Glass Transition Temperature (Tg)',
        yaxis_title='Dielectric Constant (Er)',
        coloraxis_colorbar=dict(title='Er'),
        hovermode='closest'
    )

    # Show the plot
    fig.show()

# You can call these functions to read and process the file, and create visualizations
try:
    prediction_data = read_prediction_smiles()
    
    # Create and show the scatter plot
    create_scatter_plot(prediction_data)
    
    # Uncomment these if you still want to show the other plots
    # draw_parallel_coordinates(prediction_data)
    # create_heatmap(prediction_data)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the file is in the correct location or provide the full path to the file.")
except Exception as e:
    print(f"An error occurred: {e}")
