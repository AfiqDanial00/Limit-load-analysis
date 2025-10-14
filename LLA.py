import streamlit as st
import numpy as np
import pandas as pd

# [Keep all the calculation functions the same as above...]

def create_streamlit_plot(displacements, pressures, results, method):
    """Create visualization using Streamlit's native charts"""
    
    # Create main data frame
    df = pd.DataFrame({
        'Displacement': displacements,
        'Pressure': pressures,
        'Series': 'Experimental Data'
    })
    
    # Create additional series for analysis lines
    plot_data = [df]
    
    # Elastic line
    D_max = max(displacements)
    elastic_df = pd.DataFrame({
        'Displacement': [0, D_max],
        'Pressure': [0, results["S_Elastic"] * D_max],
        'Series': 'Elastic Slope'
    })
    plot_data.append(elastic_df)
    
    if method == "TES":
        # TES line
        tes_df = pd.DataFrame({
            'Displacement': [0, D_max],
            'Pressure': [0, 0.5 * results["S_Elastic"] * D_max],
            'Series': 'TES Line'
        })
        plot_data.append(tes_df)
        
    elif method == "TI" and results.get("S_Plastic") is not None:
        # Plastic tangent
        plastic_df = pd.DataFrame({
            'Displacement': [0, D_max],
            'Pressure': [results["C_Plastic"], results["S_Plastic"] * D_max + results["C_Plastic"]],
            'Series': 'Plastic Tangent'
        })
        plot_data.append(plastic_df)
    
    # Combine all data
    combined_df = pd.concat(plot_data, ignore_index=True)
    
    # Create chart
    chart = st.line_chart(
        combined_df, 
        x='Displacement', 
        y='Pressure', 
        color='Series',
        height=500
    )
    
    # Add limit load point if found
    if results.get("is_found"):
        limit_df = pd.DataFrame({
            'Displacement': [results["D_Limit"]],
            'Pressure': [results["P_Limit"]],
            'Series': 'Limit Load Point'
        })
        st.scatter_chart(
            limit_df,
            x='Displacement',
            y='Pressure',
            color='Series',
            size=100
        )

# [Rest of the code remains the same, just replace create_plot with create_streamlit_plot]
