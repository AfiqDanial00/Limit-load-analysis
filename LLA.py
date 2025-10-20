import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

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

def find_tes_intersection(displacements, pressures, tes_slope):
    """Find intersection between TES line and experimental data using interpolation"""
    # Create fine interpolation for accurate intersection detection
    fine_d = np.linspace(min(displacements), max(displacements), 1000)
    fine_p = np.interp(fine_d, displacements, pressures)
    fine_tes_p = tes_slope * fine_d
    
    # Find where experimental data crosses above TES line
    for i in range(1, len(fine_d)):
        p_exp = fine_p[i]
        p_tes = fine_tes_p[i]
        p_exp_prev = fine_p[i-1]
        p_tes_prev = fine_tes_p[i-1]
        
        # Check if lines cross between i-1 and i
        if (p_exp_prev <= p_tes_prev and p_exp >= p_tes) or (p_exp_prev >= p_tes_prev and p_exp <= p_tes):
            # Linear interpolation to find exact intersection
            d1, d2 = fine_d[i-1], fine_d[i]
            p_exp1, p_exp2 = fine_p[i-1], fine_p[i]
            p_tes1, p_tes2 = fine_tes_p[i-1], fine_tes_p[i]
            
            # Solve for intersection: p_exp(d) = p_tes(d)
            # p_exp1 + m_exp*(d - d1) = tes_slope * d
            # where m_exp = (p_exp2 - p_exp1)/(d2 - d1)
            
            m_exp = (p_exp2 - p_exp1) / (d2 - d1) if (d2 - d1) != 0 else 0
            
            # tes_slope * d = p_exp1 + m_exp*(d - d1)
            # tes_slope * d = p_exp1 + m_exp*d - m_exp*d1
            # tes_slope * d - m_exp*d = p_exp1 - m_exp*d1
            # d*(tes_slope - m_exp) = p_exp1 - m_exp*d1
            
            denominator = tes_slope - m_exp
            if abs(denominator) > 1e-10:  # Avoid division by zero
                d_intersect = (p_exp1 - m_exp * d1) / denominator
                p_intersect = tes_slope * d_intersect
                
                # Verify the intersection is within the segment and not at origin
                if (min(d1, d2) <= d_intersect <= max(d1, d2) and 
                    d_intersect > 0.1 and p_intersect > 0.1):  # Avoid origin
                    return p_intersect, d_intersect, True
    
    return None, None, False

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
        
        # Use improved intersection detection
        P_Limit, D_Limit, is_found = find_tes_intersection(displacements, pressures, S_TES)
        
        results["P_Limit"] = P_Limit
        results["D_Limit"] = D_Limit
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
            if abs(denominator) > 1e-10:
                D_Limit = C_Plastic / denominator
                P_Limit = S_Elastic * D_Limit
                
                # Ensure intersection is not at origin and is reasonable
                if D_Limit > 0.1 and P_Limit > 0.1 and D_Limit <= max(displacements):
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
        else:
            results["P_Limit"] = None
            results["D_Limit"] = None
            results["is_found"] = False
    
    return results

def create_tes_interception_chart(displacements, pressures, results):
    """Create specialized chart for TES method showing TES line and data intersection"""
    
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
    
    # Elastic line
    elastic_pressures = results["S_Elastic"] * dense_displacements
    elastic_df = pd.DataFrame({
        'Displacement': dense_displacements,
        'Pressure': elastic_pressures,
        'Series': 'Elastic Slope',
        'Line_Type': 'Reference'
    })
    chart_data.append(elastic_df)
    
    # TES line (RED LINE - this is the important one)
    tes_pressures = 0.5 * results["S_Elastic"] * dense_displacements
    tes_df = pd.DataFrame({
        'Displacement': dense_displacements,
        'Pressure': tes_pressures,
        'Series': 'TES Line (0.5 × Elastic Slope)',
        'Line_Type': 'TES'
    })
    chart_data.append(tes_df)
    
    # Combine all data
    combined_data = pd.concat(chart_data, ignore_index=True)
    
    # Create the main chart with TES line highlighted
    line_chart = alt.Chart(combined_data).mark_line().encode(
        x=alt.X('Displacement:Q', title='Displacement', scale=alt.Scale(zero=False)),
        y=alt.Y('Pressure:Q', title='Pressure', scale=alt.Scale(zero=False)),
        color=alt.Color('Series:N', legend=alt.Legend(title="Lines"),
                       scale=alt.Scale(
                           domain=['Experimental Data', 'Elastic Slope', 'TES Line (0.5 × Elastic Slope)'],
                           range=['blue', 'green', 'red']
                       )),
        strokeWidth=alt.condition(
            alt.datum.Series == 'TES Line (0.5 × Elastic Slope)',
            alt.value(3),  # Thicker line for TES
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
        title="TES Method - Intersection of TES Line (Red) with Experimental Data"
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
    
    # Add TES intersection point if found (and not at origin)
    if results.get("is_found") and results["D_Limit"] is not None and results["P_Limit"] is not None:
        intersection_data = pd.DataFrame({
            'Displacement': [results["D_Limit"]],
            'Pressure': [results["P_Limit"]],
            'Series': ['TES Intersection Point'],
            'Description': [f'TES Intersection: D={results["D_Limit"]:.4f}, P={results["P_Limit"]:.4f}']
        })
        
        # Highlight the intersection point
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
        # Show manual intersection detection help
        st.warning("""
        ⚠️ **Auto intersection not found. Here's how to find it manually:**
        1. Look at the plot where the **red TES line** crosses the **blue experimental data line**
        2. Move your cursor over that intersection area to see coordinates
        3. The intersection should be where the lines cross (not at origin)
        4. Try adjusting the **Number of Elastic Points** in the sidebar
        """)
    
    return final_chart

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
    
    # Elastic line
    elastic_pressures = results["S_Elastic"] * dense_displacements
    elastic_df = pd.DataFrame({
        'Displacement': dense_displacements,
        'Pressure': elastic_pressures,
        'Series': 'Elastic Slope',
        'Line_Type': 'Reference'
    })
    chart_data.append(elastic_df)
    
    # Plastic tangent line (only if available)
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
        
        # Highlight the intersection point
        intersection_point = alt.Chart(intersection_data).mark_point(
            size=300,
            color='orange',
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
            color='orange',
            strokeDash=[5, 5],
            opacity=0.6
        ).encode(
            x='Displacement:Q',
            tooltip=[alt.Tooltip('Displacement:Q', title='Intersection D', format='.4f')]
        )
        
        h_line = alt.Chart(intersection_data).mark_rule(
            color='orange',
            strokeDash=[5, 5],
            opacity=0.6
        ).encode(
            y='Pressure:Q',
            tooltip=[alt.Tooltip('Pressure:Q', title='Intersection P', format='.4f')]
        )
        
        final_chart = line_chart + points + intersection_point + v_line + h_line
        
    else:
        final_chart = line_chart + points
        if results.get("S_Plastic") is None:
            st.warning("""
            ⚠️ **Plastic tangent could not be calculated.**
            - Make sure you have enough **Plastic Points** in the plastic region
            - The plastic region should be the flat part of your curve
            - Try adjusting the **Number of Plastic Points** in the sidebar
            """)
        else:
            st.warning("""
            ⚠️ **Tangent intersection not found.**
            - The elastic and plastic tangents may be parallel
            - Or the intersection point is outside the data range
            - Try adjusting the **Number of Elastic Points** or **Plastic Points**
            """)
    
    return final_chart

# --- Streamlit UI ---
def main():
    st.title("🔬 Limit Load Analyzer")
    st.markdown("**TES (Twice Elastic Slope) & TI (Tangent Intersection) Methods**")
    
    # Instructions
    st.success("""
    **🎯 BOTH TES AND TI METHODS AVAILABLE!**
    - **TES**: Red line intersection with experimental data
    - **TI**: Orange line intersection between elastic and plastic tangents
    - **Move cursor freely** to explore coordinates
    """)
    
    # Sidebar for inputs
    st.sidebar.header("⚙️ Analysis Parameters")
    
    method = st.sidebar.selectbox("Analysis Method", ["TES", "TI"])
    elastic_points = st.sidebar.number_input(
        "Number of Elastic Points", 
        min_value=2, 
        value=5,
        help="Number of initial points used for elastic slope calculation",
        key="elastic_points"
    )
    
    if method == "TI":
        plastic_points = st.sidebar.number_input(
            "Number of Plastic Points", 
            min_value=2, 
            value=5,
            help="Number of final points used for plastic tangent calculation",
            key="plastic_points"
        )
    else:
        plastic_points = 5
    
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
                if results.get("is_found") and results["D_Limit"] is not None and results["P_Limit"] is not None:
                    st.metric("Limit Pressure", f"{results['P_Limit']:.4f}")
                    st.metric("Limit Displacement", f"{results['D_Limit']:.4f}")
                else:
                    st.metric("Limit Pressure", "Not Found")
                    st.metric("Status", "❌")
            
            # Plot with the correct method
            st.subheader("📊 Interactive Plot")
            
            try:
                if method == "TES":
                    st.markdown("""
                    **🔍 TES Method - Looking for Intersection:**
                    - **Red TES Line** = 0.5 × Elastic Slope
                    - **Intersection** = Where red line crosses blue experimental data
                    """)
                    chart = create_tes_interception_chart(displacements, pressures, results)
                    
                else:  # TI method
                    st.markdown("""
                    **🔍 TI Method - Looking for Intersection:**
                    - **Green Line** = Elastic Slope from initial points
                    - **Orange Line** = Plastic Tangent from final points  
                    - **Intersection** = Where green and orange lines cross
                    """)
                    chart = create_ti_interception_chart(displacements, pressures, results)
                
                st.altair_chart(chart, use_container_width=True)
                
                # Show intersection details if found
                if results.get("is_found") and results["D_Limit"] is not None and results["P_Limit"] is not None:
                    if method == "TES":
                        st.success(f"🎯 **TES INTERSECTION FOUND!**")
                        st.info(f"""
                        **Intersection Coordinates:**
                        - **Displacement**: {results['D_Limit']:.4f}
                        - **Pressure**: {results['P_Limit']:.4f}
                        - **TES Slope**: {results.get('S_TES', 0):.4f}
                        """)
                    else:
                        st.success(f"🎯 **TANGENT INTERSECTION FOUND!**")
                        st.info(f"""
                        **Intersection Coordinates:**
                        - **Displacement**: {results['D_Limit']:.4f}
                        - **Pressure**: {results['P_Limit']:.4f}
                        - **Elastic Slope**: {results['S_Elastic']:.4f}
                        - **Plastic Slope**: {results.get('S_Plastic', 0):.4f}
                        """)
                
            except Exception as e:
                st.error(f"Error creating interactive chart: {e}")
                st.info("Please make sure Altair is installed: `pip install altair`")

if __name__ == "__main__":
    main()
