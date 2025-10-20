import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# --- App Configuration ---
st.set_page_config(
    page_title="Limit Load Analyzer (TI Method)",
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

def run_ti_analysis(displacements, pressures, elastic_points, plastic_points):
    """
    Calculates Limit Load using TI (Tangent Intersection) method.
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

    # 2. Determine Plastic Tangent (S_Plastic, C_Plastic)
    # Use the last 'plastic_points' for linear regression for the plastic tangent
    N_P = min(plastic_points, N - 1)
    d_plastic = displacements[-N_P:]
    p_plastic = pressures[-N_P:]
    
    S_Plastic, C_Plastic = calculate_linear_fit(d_plastic, p_plastic)
    results["S_Plastic"] = S_Plastic
    results["C_Plastic"] = C_Plastic

    # 3. Find intersection between elastic and plastic lines
    if S_Plastic is not None and S_Elastic is not None:
        
        # Intersection point: S_Elastic * D = S_Plastic * D + C_Plastic
        # D * (S_Elastic - S_Plastic) = C_Plastic
        
        denominator = S_Elastic - S_Plastic
        if abs(denominator) > 1e-10:  # IMPROVED: Better division check
            D_Limit = C_Plastic / denominator
            P_Limit = S_Elastic * D_Limit
            
            # IMPROVED: Enhanced validation checks
            if (D_Limit > 0.1 and P_Limit > 0.1 and 
                D_Limit <= max(displacements) and 
                D_Limit >= min(displacements) and
                P_Limit <= max(pressures) and
                P_Limit >= min(pressures)):
                results["P_Limit"] = P_Limit
                results["D_Limit"] = D_Limit
                results["is_found"] = True
            else:
                # Intersection found but outside reasonable bounds
                results["P_Limit"] = None
                results["D_Limit"] = None
                results["is_found"] = False
                results["intersection_out_of_bounds"] = True
        else:
            # Lines are parallel, no intersection
            results["P_Limit"] = None
            results["D_Limit"] = None
            results["is_found"] = False
            results["parallel_lines"] = True
    else:
        results["P_Limit"] = None
        results["D_Limit"] = None
        results["is_found"] = False
    
    return results

def create_ti_interception_chart(displacements, pressures, results):
    """Create specialized chart for TI method showing elastic and plastic tangent intersection"""
    
    # Create dense data for smooth cursor movement
    D_max = max(displacements)
    dense_displacements = np.linspace(0, D_max, 500)
    
    # Create data for all lines
    chart_data = []
    
    # Experimental data (interpolated for smoothness)
    exp_pressures = np.interp(dense_displacements, displacements, pressures)
    exp_df = pd.DataFrame({
        'Displacement': dense_displacements,
        'Pressure': exp_pressures,
        'Series': 'Experimental Data',
        'Line_Type': 'Curve'
    })
    chart_data.append(exp_df)
    
    # Elastic line (GREEN)
    elastic_pressures = results["S_Elastic"] * dense_displacements
    elastic_df = pd.DataFrame({
        'Displacement': dense_displacements,
        'Pressure': elastic_pressures,
        'Series': 'Elastic Slope',
        'Line_Type': 'Reference'
    })
    chart_data.append(elastic_df)
    
    # Plastic tangent (ORANGE) - only if available
    if results.get("S_Plastic") is not None:
        plastic_pressures = results["S_Plastic"] * dense_displacements + results["C_Plastic"]
        plastic_df = pd.DataFrame({
            'Displacement': dense_displacements,
            'Pressure': plastic_pressures,
            'Series': 'Plastic Tangent',
            'Line_Type': 'Reference'
        })
        chart_data.append(plastic_df)
    
    # Combine all data
    combined_data = pd.concat(chart_data, ignore_index=True)
    
    # Create the main chart
    line_chart = alt.Chart(combined_data).mark_line().encode(
        x=alt.X('Displacement:Q', title='Displacement', scale=alt.Scale(zero=False)),
        y=alt.Y('Pressure:Q', title='Pressure', scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N', legend=alt.Legend(title="Lines"),
                       scale=alt.Scale(
                           domain=['Experimental Data', 'Elastic Slope', 'Plastic Tangent'],
                           range=['blue', 'green', 'orange']
                       )),
        strokeWidth=alt.condition(
            alt.datum.Series == 'Plastic Tangent',
            alt.value(3),  # Thicker line for plastic tangent
            alt.value(2)   # Normal thickness for others
        ),
        strokeDash=alt.condition(
            alt.datum.Series == 'Experimental Data',
            alt.value([0]),  # solid for experimental
            alt.value([5, 5])  # dashed for reference lines
        ),
        tooltip=[
            alt.Tooltip('Series:N', title='Line'),
            alt.Tooltip('Displacement:Q', title='Displacement', format='.4f'),
            alt.Tooltip('Pressure:Q', title='Pressure', format='.4f')
        ]
    ).properties(
        width=800,
        height=500,
        title="TI Method - Intersection of Elastic Slope and Plastic Tangent"
    ).interactive()
    
    # Add original data points as circles
    points_data = pd.DataFrame({
        'Displacement': displacements,
        'Pressure': pressures,
        'Series': 'Data Points'
    })
    
    points = alt.Chart(points_data).mark_circle(
        size=60,
        color='blue',
        opacity=0.6
    ).encode(
        x='Displacement:Q',
        y='Pressure:Q',
        tooltip=[
            alt.Tooltip('Displacement:Q', title='Displacement', format='.4f'),
            alt.Tooltip('Pressure:Q', title='Pressure', format='.4f')
        ]
    )
    
    # Add tangent intersection point if found
    if results.get("is_found") and results["D_Limit"] is not None and results["P_Limit"] is not None:
        intersection_data = pd.DataFrame({
            'Displacement': [results["D_Limit"]],
            'Pressure': [results["P_Limit"]],
            'Series': ['Tangent Intersection Point'],
            'Description': [f'TI Intersection: D={results["D_Limit"]:.4f}, P={results["P_Limit"]:.4f}']
        })
        
        # Highlight the intersection point with diamond
        intersection_point = alt.Chart(intersection_data).mark_point(
            size=300,
            color='red',
            shape='diamond',
            filled=True,
            opacity=1.0
        ).encode(
            x='Displacement:Q',
            y='Pressure:Q',
            tooltip=[
                alt.Tooltip('Description:N', title='Intersection')
            ]
        )
        
        # Add vertical and horizontal lines to highlight intersection
        v_line = alt.Chart(intersection_data).mark_rule(
            color='red',
            strokeDash=[5, 5],
            opacity=0.6
        ).encode(
            x='Displacement:Q',
            tooltip=[alt.Tooltip('Displacement:Q', title='Intersection D', format='.4f')]
        )
        
        h_line = alt.Chart(intersection_data).mark_rule(
            color='red',
            strokeDash=[5, 5],
            opacity=0.6
        ).encode(
            y='Pressure:Q',
            tooltip=[alt.Tooltip('Pressure:Q', title='Intersection P', format='.4f')]
        )
        
        final_chart = line_chart + points + intersection_point + v_line + h_line
        
    else:
        final_chart = line_chart + points
        
        # IMPROVED: More specific error messages
        if results.get("S_Plastic") is None:
            st.warning("""
            ⚠️ **Plastic tangent could not be calculated.**
            - Make sure you have enough **Plastic Points** in the plastic region
            - The plastic region should be the flat part of your curve
            - Try increasing the **Number of Plastic Points** in the sidebar
            - Current plastic points used: {min(plastic_points, len(displacements) - 1)}
            """)
        elif results.get("parallel_lines"):
            st.warning("""
            ⚠️ **Elastic and plastic lines are parallel.**
            - The slopes are too similar (S_elastic ≈ S_plastic)
            - Try adjusting the **Number of Elastic Points** or **Plastic Points**
            - Elastic slope: {results['S_Elastic']:.4f}
            - Plastic slope: {results.get('S_Plastic', 0):.4f}
            """)
        elif results.get("intersection_out_of_bounds"):
            st.warning("""
            ⚠️ **Intersection found but outside data range.**
            - The calculated intersection is not within your data boundaries
            - This can happen if the plastic region is not well-defined
            - Try adjusting the **Number of Plastic Points**
            """)
        elif not results.get("is_found"):
            st.warning("""
            ⚠️ **Tangent intersection not found.**
            - Check that both elastic and plastic regions are properly defined
            - Try adjusting the **Number of Elastic Points** or **Plastic Points**
            - Ensure your data has clear elastic and plastic regions
            """)
    
    return final_chart

# --- Streamlit UI ---
def main():
    st.title("🔬 Limit Load Analyzer - TI Method")
    st.markdown("**Tangent Intersection (TI) Method**")
    
    # Instructions
    st.success("""
    **🎯 IMPROVED TI METHOD - Enhanced Intersection Detection**
    - **Better validation**: Ensures intersections are within data bounds
    - **Parallel line detection**: Identifies when slopes are too similar
    - **Origin avoidance**: Prevents false intersections at (0,0)
    - **Interactive visualization**: Move cursor freely to explore coordinates
    """)
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ TI Analysis Parameters")
    
    elastic_points = st.sidebar.number_input(
        "Number of Elastic Points", 
        min_value=2, 
        value=5,
        help="Number of initial points used for elastic slope calculation"
    )
    
    plastic_points = st.sidebar.number_input(
        "Number of Plastic Points", 
        min_value=2, 
        value=5,
        help="Number of final points used for plastic tangent calculation"
    )
    
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
    if st.sidebar.button("🚀 Run TI Analysis", type="primary"):
        with st.spinner("Analyzing data with enhanced TI method..."):
            # Perform TI analysis with improved algorithm
            results = run_ti_analysis(displacements, pressures, elastic_points, plastic_points)
            
            # Display results
            st.header("📈 Enhanced TI Analysis Results")
            
            # Key metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Elastic Slope", f"{results['S_Elastic']:.4f}")
                st.write(f"P = {results['S_Elastic']:.4f} × D")
            with col2:
                if results.get("S_Plastic") is not None:
                    st.metric("Plastic Slope", f"{results.get('S_Plastic', 0):.4f}")
                    st.write(f"P = {results.get('S_Plastic', 0):.4f} × D + {results.get('C_Plastic', 0):.4f}")
                else:
                    st.metric("Plastic Slope", "N/A")
            with col3:
                if results.get("is_found"):
                    st.metric("Limit Pressure", f"{results['P_Limit']:.4f}")
                    st.metric("Limit Displacement", f"{results['D_Limit']:.4f}")
                else:
                    st.metric("Limit Pressure", "Not Found")
                    st.metric("Status", "❌")
            
            # Plot with enhanced TI interception display
            st.subheader("📊 Enhanced TI Method - Tangent Intersection Plot")
            st.markdown("""
            **🔍 Enhanced TI Method Features:**
            - **Improved intersection detection** with better validation
            - **Parallel line identification** with specific warnings
            - **Bounds checking** to ensure realistic intersections
            - **Interactive exploration** with free cursor movement
            """)
            
            try:
                chart = create_ti_interception_chart(displacements, pressures, results)
                st.altair_chart(chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating interactive chart: {e}")
                st.info("Please make sure Altair is installed: `pip install altair`")
            
            # Show detailed intersection results if found
            if results.get("is_found"):
                st.success(f"🎯 **VALID TANGENT INTERSECTION FOUND!**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **Validated Intersection Coordinates:**
                    - **Displacement**: {results['D_Limit']:.6f}
                    - **Pressure**: {results['P_Limit']:.6f}
                    - **Status**: ✅ Within data bounds
                    """)
                
                with col2:
                    st.info(f"""
                    **Validated Line Equations:**
                    - **Elastic**: P = {results['S_Elastic']:.4f} × D
                    - **Plastic**: P = {results.get('S_Plastic', 0):.4f} × D + {results.get('C_Plastic', 0):.4f}
                    - **Slope Difference**: {abs(results['S_Elastic'] - results.get('S_Plastic', 0)):.6f}
                    """)
            
            # Show enhanced calculation details
            with st.expander("🔍 View Enhanced Calculation Details"):
                st.write("**Elastic Region Analysis:**")
                st.write(f"- Points used: {min(elastic_points, len(displacements) - 1)}")
                st.write(f"- Displacement range: {displacements[1]:.3f} to {displacements[min(elastic_points, len(displacements)-1)]:.3f}")
                st.write(f"- Slope calculation: Through origin regression")
                st.write(f"- Validation: Slope = {results['S_Elastic']:.6f}")
                
                if results.get("S_Plastic") is not None:
                    st.write("**Plastic Region Analysis:**")
                    st.write(f"- Points used: {min(plastic_points, len(displacements) - 1)}")
                    st.write(f"- Displacement range: {displacements[-plastic_points]:.3f} to {displacements[-1]:.3f}")
                    st.write(f"- Slope calculation: Linear regression")
                    st.write(f"- Validation: Slope = {results.get('S_Plastic', 0):.6f}, Intercept = {results.get('C_Plastic', 0):.6f}")
                
                if results.get("is_found"):
                    st.write("**Enhanced Intersection Validation:**")
                    st.write(f"- ✅ Not at origin (D > 0.1, P > 0.1)")
                    st.write(f"- ✅ Within displacement bounds: {min(displacements):.3f} ≤ {results['D_Limit']:.3f} ≤ {max(displacements):.3f}")
                    st.write(f"- ✅ Within pressure bounds: {min(pressures):.3f} ≤ {results['P_Limit']:.3f} ≤ {max(pressures):.3f}")
                    st.write(f"- ✅ Non-parallel lines: |S_elastic - S_plastic| = {abs(results['S_Elastic'] - results.get('S_Plastic', 0)):.6f} > 1e-10")

if __name__ == "__main__":
    main()
