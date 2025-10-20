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

def create_interactive_plot_with_free_cursor(displacements, pressures, results, method):
    """Create visualization with manual coordinate exploration"""
    
    # Create comprehensive data frame with all series
    plot_data = []
    
    # Experimental data
    exp_df = pd.DataFrame({
        'Displacement': displacements,
        'Pressure': pressures,
        'Series': 'Experimental Data',
        'Point_Type': 'Data'
    })
    plot_data.append(exp_df)
    
    # Generate points for elastic line
    D_max = max(displacements)
    elastic_displacements = np.linspace(0, D_max, 100)
    elastic_pressures = results["S_Elastic"] * elastic_displacements
    
    elastic_df = pd.DataFrame({
        'Displacement': elastic_displacements,
        'Pressure': elastic_pressures,
        'Series': 'Elastic Slope',
        'Point_Type': 'Line'
    })
    plot_data.append(elastic_df)
    
    if method == "TES":
        # Generate points for TES line
        tes_pressures = 0.5 * results["S_Elastic"] * elastic_displacements
        
        tes_df = pd.DataFrame({
            'Displacement': elastic_displacements,
            'Pressure': tes_pressures,
            'Series': 'TES Line (0.5 × Elastic Slope)',
            'Point_Type': 'Line'
        })
        plot_data.append(tes_df)
        
    elif method == "TI" and results.get("S_Plastic") is not None:
        # Generate points for plastic tangent
        plastic_pressures = results["S_Plastic"] * elastic_displacements + results["C_Plastic"]
        
        plastic_df = pd.DataFrame({
            'Displacement': elastic_displacements,
            'Pressure': plastic_pressures,
            'Series': 'Plastic Tangent',
            'Point_Type': 'Line'
        })
        plot_data.append(plastic_df)
    
    # Combine all data
    combined_df = pd.concat(plot_data, ignore_index=True)
    
    # --- MANUAL COORDINATE EXPLORER ---
    st.markdown("---")
    st.subheader("🎯 Free Coordinate Explorer")
    st.info("**Move the slider to explore any point along the lines freely - NO AUTO-SNAP!**")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        manual_d = st.slider(
            "Displacement Position",
            min_value=float(0),
            max_value=float(max(displacements)),
            value=float(max(displacements) / 2),
            step=0.001,
            format="%.4f",
            key="manual_displacement"
        )
    
    with col2:
        # Calculate corresponding pressure values for manual displacement
        elastic_p = results["S_Elastic"] * manual_d
        experimental_p = np.interp(manual_d, displacements, pressures)
        
        st.metric("Current Position", f"D = {manual_d:.4f}")
        st.metric("Experimental Data Pressure", f"{experimental_p:.4f}")
        st.metric("Elastic Line Pressure", f"{elastic_p:.4f}")
        
        if method == "TES":
            tes_p = 0.5 * results["S_Elastic"] * manual_d
            st.metric("TES Line Pressure", f"{tes_p:.4f}")
        elif method == "TI" and results.get("S_Plastic") is not None:
            plastic_p = results["S_Plastic"] * manual_d + results["C_Plastic"]
            st.metric("Plastic Line Pressure", f"{plastic_p:.4f}")
    
    with col3:
        st.markdown("**Line Equations:**")
        st.write(f"**Elastic:** P = {results['S_Elastic']:.4f} × D")
        if method == "TES":
            st.write(f"**TES:** P = {0.5 * results['S_Elastic']:.4f} × D")
        elif method == "TI" and results.get("S_Plastic") is not None:
            st.write(f"**Plastic:** P = {results['S_Plastic']:.4f} × D + {results['C_Plastic']:.4f}")
    
    # Create the main chart with the current manual position
    st.markdown("---")
    st.subheader("📊 Analysis Plot")
    
    # Add manual point to the data
    manual_point_df = pd.DataFrame({
        'Displacement': [manual_d],
        'Pressure': [experimental_p],
        'Series': ['Current Position'],
        'Point_Type': ['Manual']
    })
    
    all_data_df = pd.concat([combined_df, manual_point_df], ignore_index=True)
    
    # Create the main chart
    chart = st.line_chart(
        all_data_df[all_data_df['Point_Type'].isin(['Line', 'Data'])], 
        x='Displacement', 
        y='Pressure', 
        color='Series',
        height=500
    )
    
    # Add manual point as a separate scatter plot
    st.scatter_chart(
        manual_point_df,
        x='Displacement',
        y='Pressure',
        color='Series',
        size=100
    )
    
    # Add limit point as a separate scatter plot if found
    if results.get("is_found"):
        limit_points = pd.DataFrame({
            'Displacement': [results["D_Limit"]],
            'Pressure': [results["P_Limit"]],
            'Series': ['Auto Limit Load Point']
        })
        
        st.scatter_chart(
            limit_points,
            x='Displacement',
            y='Pressure',
            color='Series',
            size=100
        )
        
        st.success(f"🎯 Auto Limit Load Found at P = {results['P_Limit']:.4f}, D = {results['D_Limit']:.4f}")

# --- Streamlit UI ---
def main():
    st.title("🔬 Limit Load Analyzer")
    st.markdown("**TES (Twice Elastic Slope) & TI (Tangent Intersection) Methods**")
    
    # Instructions for new cursor functionality
    st.success("""
    **🎯 NEW: True Free Cursor Movement!**
    - **No auto-snapping** to data points
    - **Move the slider freely** to explore any coordinate
    - **See exact values** on all lines simultaneously
    - **Continuous exploration** - not limited to discrete points
    """)
    
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
                    st.metric("Auto Limit Pressure", f"{results['P_Limit']:.4f}")
                else:
                    st.metric("Auto Limit Pressure", "Not Found")
            
            # Plot results with FREE cursor movement
            create_interactive_plot_with_free_cursor(displacements, pressures, results, method)
            
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
                        st.success("✅ Auto limit load successfully determined")
                        st.write(f"- Auto Limit Pressure: {results['P_Limit']:.6f}")
                        st.write(f"- Auto Limit Displacement: {results['D_Limit']:.6f}")
                    else:
                        st.error("❌ Auto limit load not found with current parameters")
                        
                elif method == "TI":
                    st.write("**TI Method Results:**")
                    st.write(f"- Plastic Slope: {results.get('S_Plastic', 0):.6f}")
                    st.write(f"- Plastic Intercept: {results.get('C_Plastic', 0):.6f}")
                    st.write(f"- Plastic line equation: P = {results.get('S_Plastic', 0):.4f} × D + {results.get('C_Plastic', 0):.4f}")
                    if results.get("is_found"):
                        st.success("✅ Auto limit load successfully determined")
                        st.write(f"- Auto Limit Pressure: {results['P_Limit']:.6f}")
                        st.write(f"- Auto Limit Displacement: {results['D_Limit']:.6f}")
                    else:
                        st.error("❌ Auto limit load not found with current parameters")
            
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
Plastic Line Equation: P = {results.get('S_Plastic', 0):4f} × D + {results.get('C_Plastic', 0):.4f}
"""
                
                if results.get("is_found"):
                    results_text += f"""
AUTO LIMIT LOAD RESULTS:
Limit Pressure: {results['P_Limit']:.6f}
Limit Displacement: {results['D_Limit']:.6f}
"""
                else:
                    results_text += "\nAuto limit load not found with current parameters."
                
                st.download_button(
                    label="📥 Download Results as Text",
                    data=results_text,
                    file_name=f"limit_load_analysis_{method}.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
