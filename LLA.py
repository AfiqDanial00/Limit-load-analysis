import streamlit as st
import numpy as np
import pandas as pd

# --- App Configuration ---
st.set_page_config(
    page_title="Limit Load Analyzer (TES & TI)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Core Calculation Functions ---

def parse_data(data_text):
    """Parses raw text input into a numpy array of floats, excluding (0,0) if it's the only line."""
    try:
        data = np.array([float(line.strip()) for line in data_text.split('\n') if line.strip()])
        return data
    except ValueError:
        st.error("Error: Please ensure all data points are valid numbers.")
        return np.array([])

def calculate_linear_fit(x_data, y_data):
    """Performs linear regression (P = S*delta + C) on provided data."""
    if len(x_data) < 2:
        return None, None # Slope (S), Intercept (C)
    
    # Simple linear regression (polyfit degree 1)
    coeffs = np.polyfit(x_data, y_data, 1)
    S = coeffs[0]
    C = coeffs[1]
    return S, C

def calculate_slope_through_origin(x_data, y_data):
    """Calculates slope for line passing through (0,0): S = sum(P*D) / sum(D^2)"""
    if np.sum(x_data**2) == 0:
        return 0
    S = np.sum(x_data * y_data) / np.sum(x_data**2)
    return S

def run_analysis(displacements, pressures, method, elastic_points, plastic_points):
    """
    Calculates Limit Load using either TES or TI method.
    The input arrays MUST include the (0,0) point at index 0.
    """
    results = {}
    N = len(displacements)
    
    # 1. Determine Elastic Slope (S_Elastic)
    # Use points from index 1 up to 'elastic_points' (inclusive) for regression
    N_E = min(elastic_points, N - 1)
    d_elastic = displacements[1:N_E + 1]
    p_elastic = pressures[1:N_E + 1]

    # Use simplified regression (line passes through origin) for the elastic line
    S_Elastic = calculate_slope_through_origin(d_elastic, p_elastic)
    
    results["S_Elastic"] = S_Elastic

    # --- Twice Elastic Slope (TES) Method ---
    if method == "TES":
        S_TES = 0.5 * S_Elastic
        results["S_TES"] = S_TES
        
        P_Limit, D_Limit = np.nan, np.nan
        is_found = False

        # Find intersection where P_actual >= S_TES * D_actual
        for i in range(1, N):
            P_actual = pressures[i]
            D_actual = displacements[i]
            P_on_TES = S_TES * D_actual

            if P_actual >= P_on_TES:
                # Interpolate intersection between point i-1 and i
                P_prev = pressures[i - 1]
                D_prev = displacements[i - 1]
                
                # Line 1 (Actual Curve segment): P = m_actual * D + c_actual
                m_actual = (P_actual - P_prev) / (D_actual - D_prev) if (D_actual - D_prev) != 0 else 0
                c_actual = P_actual - m_actual * D_actual

                # Line 2 (TES Line): P = S_TES * D
                denominator = S_TES - m_actual
                
                if denominator != 0:
                    D_Limit = c_actual / denominator
                    P_Limit = S_TES * D_Limit
                    is_found = True
                break
        
        results["P_Limit"] = P_Limit if is_found else None
        results["D_Limit"] = D_Limit if is_found else None
        results["is_found"] = is_found
        
    # --- Tangent Intersection (TI) Method ---
    elif method == "TI":
        
        # 2. Determine Plastic Tangent (S_Plastic, C_Plastic)
        # Use the last 'plastic_points' for linear regression for the plastic tangent
        N_P = min(plastic_points, N - 1)
        d_plastic = displacements[-N_P:]
        p_plastic = pressures[-N_P:]
        
        S_Plastic, C_Plastic = calculate_linear_fit(d_plastic, p_plastic)
        results["S_Plastic"] = S_Plastic
        results["C_Plastic"] = C_Plastic

        if S_Plastic is not None and S_Elastic is not None:
            
            # Intersection point: S_Elastic * D = S_Plastic * D + C_Plastic
            # D * (S_Elastic - S_Plastic) = C_Plastic
            
            denominator = S_Elastic - S_Plastic
            if denominator != 0:
                D_Limit = C_Plastic / denominator
                P_Limit = S_Elastic * D_Limit
                results["P_Limit"] = P_Limit
                results["D_Limit"] = D_Limit
                results["is_found"] = True
            else:
                results["P_Limit"] = None
                results["D_Limit"] = None
                results["is_found"] = False
        else:
            results["P_Limit"] = None
            results["D_Limit"] = None
            results["is_found"] = False
    
    return results

def create_streamlit_plot(displacements, pressures, results, method):
    """Create visualization using Streamlit's native charts"""
    
    # Create main data frame for experimental data
    main_df = pd.DataFrame({
        'Displacement': displacements,
        'Pressure': pressures,
        'Series': 'Experimental Data'
    })
    
    # Create plot data list
    plot_data = [main_df]
    
    # Generate points for elastic line
    D_max = max(displacements)
    elastic_displacements = np.linspace(0, D_max, 50)
    elastic_pressures = results["S_Elastic"] * elastic_displacements
    
    elastic_df = pd.DataFrame({
        'Displacement': elastic_displacements,
        'Pressure': elastic_pressures,
        'Series': 'Elastic Slope'
    })
    plot_data.append(elastic_df)
    
    if method == "TES":
        # Generate points for TES line
        tes_pressures = 0.5 * results["S_Elastic"] * elastic_displacements
        
        tes_df = pd.DataFrame({
            'Displacement': elastic_displacements,
            'Pressure': tes_pressures,
            'Series': 'TES Line (0.5 × Elastic Slope)'
        })
        plot_data.append(tes_df)
        
    elif method == "TI" and results.get("S_Plastic") is not None:
        # Generate points for plastic tangent
        plastic_pressures = results["S_Plastic"] * elastic_displacements + results["C_Plastic"]
        
        plastic_df = pd.DataFrame({
            'Displacement': elastic_displacements,
            'Pressure': plastic_pressures,
            'Series': 'Plastic Tangent'
        })
        plot_data.append(plastic_df)
    
    # Combine all data
    combined_df = pd.concat(plot_data, ignore_index=True)
    
    # Create the main chart
    st.line_chart(
        combined_df, 
        x='Displacement', 
        y='Pressure', 
        color='Series',
        height=500
    )
    
    # Add limit load point if found
    if results.get("is_found"):
        st.success(f"🎯 Limit Load Found at P = {results['P_Limit']:.4f}, D = {results['D_Limit']:.4f}")
        
        # Create a separate chart for the limit point
        limit_df = pd.DataFrame({
            'Displacement': [results["D_Limit"]],
            'Pressure': [results["P_Limit"]],
            'Series': 'Limit Load Point'
        })
        
        # Use scatter chart for the limit point
        st.scatter_chart(
            limit_df,
            x='Displacement',
            y='Pressure',
            color='Series',
            size=100
        )

# --- Streamlit UI ---
def main():
    st.title("🔬 Limit Load Analyzer")
    st.markdown("**TES (Twice Elastic Slope) & TI (Tangent Intersection) Methods**")
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Analysis Parameters")
    
    method = st.sidebar.selectbox("Analysis Method", ["TES", "TI"])
    elastic_points = st.sidebar.number_input(
        "Number of Elastic Points", 
        min_value=2, 
        value=5,
        help="Number of initial points used for elastic slope calculation"
    )
    
    if method == "TI":
        plastic_points = st.sidebar.number_input(
            "Number of Plastic Points", 
            min_value=2, 
            value=5,
            help="Number of final points used for plastic tangent calculation"
        )
    else:
        plastic_points = 5  # Not used for TES, but needed for function call
    
    # Data input section
    st.sidebar.header("📊 Input Data")
    st.sidebar.markdown("Enter data points (one per line), **starting with (0,0)**:")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        displacement_text = st.text_area(
            "Displacement", 
            "0.0\n0.5\n1.0\n1.5\n2.0\n2.5\n3.0\n3.5\n4.0\n4.5\n5.0", 
            height=200
        )
    with col2:
        pressure_text = st.text_area(
            "Pressure", 
            "0.0\n0.8\n1.5\n2.1\n2.6\n2.9\n3.1\n3.2\n3.25\n3.27\n3.28", 
            height=200
        )
    
    # Parse input data
    displacements = parse_data(displacement_text)
    pressures = parse_data(pressure_text)
    
    # Validation
    if len(displacements) == 0 or len(pressures) == 0:
        st.warning("⚠️ Please enter valid numerical data in both columns")
        return
        
    if len(displacements) != len(pressures):
        st.error("❌ Error: Displacement and Pressure must have the same number of data points")
        return
    
    if len(displacements) < 3:
        st.error("❌ Error: Need at least 3 data points for analysis")
        return
    
    # Run analysis when button is clicked
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Analyzing data..."):
            # Perform analysis
            results = run_analysis(displacements, pressures, method, elastic_points, plastic_points)
            
            # Display results
            st.header("📈 Analysis Results")
            
            # Key metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Elastic Slope", f"{results['S_Elastic']:.4f}")
            with col2:
                if method == "TES":
                    st.metric("TES Slope", f"{results.get('S_TES', 0):.4f}")
                elif method == "TI":
                    st.metric("Plastic Slope", f"{results.get('S_Plastic', 0):.4f}")
            with col3:
                if results.get("is_found"):
                    st.metric("Limit Pressure", f"{results['P_Limit']:.4f}")
                    st.metric("Limit Displacement", f"{results['D_Limit']:.4f}")
                else:
                    st.metric("Limit Pressure", "Not Found")
                    st.metric("Status", "❌")
            
            # Plot results
            st.subheader("📊 Analysis Plot")
            create_streamlit_plot(displacements, pressures, results, method)
            
            # Detailed results
            st.subheader("🔍 Detailed Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Elastic Region Analysis:**")
                st.write(f"- Elastic Slope (S_Elastic): {results['S_Elastic']:.6f}")
                st.write(f"- Points used for elastic slope: {min(elastic_points, len(displacements) - 1)}")
                st.write(f"- Elastic line equation: P = {results['S_Elastic']:.4f} × D")
                
            with col2:
                if method == "TES":
                    st.write("**TES Method Results:**")
                    st.write(f"- TES Slope: {results.get('S_TES', 0):.6f}")
                    st.write(f"- TES line equation: P = {results.get('S_TES', 0):.4f} × D")
                    if results.get("is_found"):
                        st.success("✅ Limit load successfully determined")
                        st.write(f"- Limit Pressure: {results['P_Limit']:.6f}")
                        st.write(f"- Limit Displacement: {results['D_Limit']:.6f}")
                    else:
                        st.error("❌ Limit load not found with current parameters")
                        
                elif method == "TI":
                    st.write("**TI Method Results:**")
                    st.write(f"- Plastic Slope: {results.get('S_Plastic', 0):.6f}")
                    st.write(f"- Plastic Intercept: {results.get('C_Plastic', 0):.6f}")
                    st.write(f"- Plastic line equation: P = {results.get('S_Plastic', 0):.4f} × D + {results.get('C_Plastic', 0):.4f}")
                    if results.get("is_found"):
                        st.success("✅ Limit load successfully determined")
                        st.write(f"- Limit Pressure: {results['P_Limit']:.6f}")
                        st.write(f"- Limit Displacement: {results['D_Limit']:.6f}")
                    else:
                        st.error("❌ Limit load not found with current parameters")
            
            # Raw data table
            with st.expander("📋 View Raw Data Table"):
                data_df = pd.DataFrame({
                    'Point': range(len(displacements)),
                    'Displacement': displacements,
                    'Pressure': pressures
                })
                st.dataframe(data_df, use_container_width=True)
                
            # Download results
            with st.expander("💾 Download Results"):
                results_text = f"""Limit Load Analysis Results - {method} Method
                
Elastic Slope (S_Elastic): {results['S_Elastic']:.6f}
Points used for elastic slope: {min(elastic_points, len(displacements) - 1)}

"""
                if method == "TES":
                    results_text += f"""TES Slope: {results.get('S_TES', 0):.6f}
TES Line Equation: P = {results.get('S_TES', 0):.4f} × D
"""
                elif method == "TI":
                    results_text += f"""Plastic Slope: {results.get('S_Plastic', 0):.6f}
Plastic Intercept: {results.get('C_Plastic', 0):.6f}
Plastic Line Equation: P = {results.get('S_Plastic', 0):.4f} × D + {results.get('C_Plastic', 0):.4f}
"""
                
                if results.get("is_found"):
                    results_text += f"""
LIMIT LOAD RESULTS:
Limit Pressure: {results['P_Limit']:.6f}
Limit Displacement: {results['D_Limit']:.6f}
"""
                else:
                    results_text += "\nLimit load not found with current parameters."
                
                st.download_button(
                    label="📥 Download Results as Text",
                    data=results_text,
                    file_name=f"limit_load_analysis_{method}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
