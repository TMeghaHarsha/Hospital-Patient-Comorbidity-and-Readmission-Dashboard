import plotly.express as px
import pandas as pd

def get_summary_statistics(df, numerical_features):
    """
    Get summary statistics for the dataframe.
    
    Args:
        df: pandas DataFrame
        numerical_features: list of numerical feature column names
        
    Returns:
        pandas DataFrame: Summary statistics
    """
    return df[numerical_features].describe()

def plot_distributions(df):
    """
    Create distribution plots for the data.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (histogram_figure, pie_chart_figure)
    """
    # Histogram of time_in_hospital
    hist_fig = px.histogram(
        df, 
        x='time_in_hospital',
        title='Distribution of Time in Hospital',
        labels={'time_in_hospital': 'Time in Hospital (days)', 'count': 'Frequency'},
        color_discrete_sequence=['#007bff']
    )
    
    # Update histogram layout with clean styling
    hist_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, color='#333333'),
        title_font_size=18,
        title_font_color='#333333',
        xaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd',
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        yaxis=dict(
            gridcolor='#f0f0f0', 
            linecolor='#ddd',
            title_font_color='#333333',
            tickfont_color='#333333'
        ),
        bargap=0.1
    )
    
    # Pie chart of readmitted values
    readmitted_counts = df['readmitted'].value_counts()
    pie_fig = px.pie(
        values=readmitted_counts.values,
        names=readmitted_counts.index,
        title='Distribution of Readmission Status',
        color_discrete_sequence=['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1']
    )
    
    # Update pie chart layout with clean styling
    pie_fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=14, color='#333333'),
        title_font_size=18,
        title_font_color='#333333',
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            font=dict(color='#333333', size=12)
        )
    )
    
    return hist_fig, pie_fig
