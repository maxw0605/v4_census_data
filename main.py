# --- Imports
import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# --- Global settings
st.set_page_config(
    page_title="Dashboard USA",
    layout="wide"
)

common_font = dict(family="Arial", size=12, color="black")

# --- Definitions
# Path
base_path = "data"
frost_path = "us_states_frost_percentage_all_days.xlsx"

#base_path = r"C:\Users\maxim\OneDrive - FH Muenster\- Masterstudium (Gebäudetechnik)\3. Semester\- Masterarbeit\Diagramme und Abbildungen\Daten\Census data"
#base_path = r"C:\Users\mwi\OneDrive - FH Muenster\- Masterstudium (Gebäudetechnik)\3. Semester\- Masterarbeit\Diagramme und Abbildungen\Daten\Census data"

state_abbr = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR',
    'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA',
    'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

frost_states2 = [
    "Alaska", "Arizona", "Colorado", "Connecticut", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas",
    "Maine", "Massachusetts", "Michigan", "Minnesota", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Dakota", "Ohio", "Oregon",
    "Pennsylvania", "Rhode Island", "South Dakota", "Utah", "Vermont", "Washington", "West Virginia",
    "Wisconsin", "Wyoming"
]

frost_states = [
    "Alaska", "Arizona", "Colorado", "Connecticut", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", 
    "Maine", "Massachusetts", "Michigan", "Minnesota", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Dakota", "Tennessee", "Utah", "Vermont", "Virginia", "Washington", 
    "West Virginia", "Wisconsin", "Wyoming"
]

df_frost = pd.read_excel(frost_path)

def get_frost_states(df, frost_day_threshold, area_threshold):
    """
    Gibt eine Liste von Bundesstaaten zurück, bei denen mindestens `area_threshold` % der Fläche
    mindestens `frost_day_threshold` Frosttage pro Jahr aufweisen.
    """
    col_name = f"percentage_over_{frost_day_threshold}_days"
    
    if col_name not in df.columns:
        raise ValueError(f"Spalte '{col_name}' existiert nicht in den Daten.")

    frost_states = df[df[col_name] >= area_threshold]["name"].tolist()
    return frost_states



# --- Titel
st.title("Dashboard USA")


# --- Staaten Auswahl
col1, col2 = st.columns(2)

with col1:
    selected_frost_day_threshold = st.number_input(
        "Lower limit of Frostdays per Year",
        min_value=1,
        max_value=365,
        value=50,
        step=1
    )

with col2:
    area_threshold = st.number_input(
        "Lower limit Percentage of States Area",
        min_value=1,
        max_value=100,
        value=50,
        step=1
    )

frost_states = get_frost_states(df_frost, selected_frost_day_threshold, area_threshold)
all_states = list(state_abbr.keys())

if "selected_states" not in st.session_state:
    st.session_state.selected_states = frost_states  # Default-Auswahl2050

# --- Auswahl-Buttons
text = (f"Frost States: More than {selected_frost_day_threshold} Frost Days in min. {area_threshold} % of the Staten Area.")

if st.button("Select Frost States"):
    st.session_state.selected_states = frost_states
    text = (f"Frost States: More than {selected_frost_day_threshold} Frost Days in min. {area_threshold} % of the Staten Area.")

if st.button("Select All States"):
    st.session_state.selected_states = all_states
    text = "Frost States: All States"

selected_states = st.multiselect(
    "Select individual states:",
    options=all_states,
    default=st.session_state.selected_states,
    key="selected_states"
)

if set(selected_states) != set(st.session_state.selected_states):
    st.session_state.selected_states = selected_states
    text = "Frost States: Individual States"

st.write(text)

selected_states = st.session_state.selected_states



# --- map
df_map_frost_states = pd.DataFrame({
    'state_name': list(state_abbr.keys()),
    'abbr': list(state_abbr.values())
})
df_map_frost_states["is_selected"] = df_map_frost_states["state_name"].apply(
    lambda x: "1" if x in selected_states else "0"
)

fig_map_selection = px.choropleth(
    df_map_frost_states,
    locations="abbr",
    locationmode="USA-states",
    color="is_selected",
    scope="usa",
    color_discrete_map={"0": "#f0f0f0", "1": "#1f77b4"},
    labels={"is_selected": "Selected"},
    hover_name="state_name",
    hover_data={"abbr": False, "is_selected": False}
)
fig_map_selection.update_layout(
    geo=dict(showlakes=False, lakecolor="lightblue"),
    dragmode=False,
    uirevision="static",
    coloraxis_showscale=False,
    showlegend=False,
    height=400,
    margin=dict(l=0, r=0, t=40, b=0)
)

col1, col2, col3 = st.columns([1, 3, 1])  # Verhältnis der Spaltenbreiten

with col2:
    st.plotly_chart(fig_map_selection, use_container_width=False)




# --- Function for Extracting statistics for a state or multiple states
def get_aggregated_stat_from_states(states, file_name, stat_name, method="sum"):
    """
    Aggregates a numeric value across multiple states from a specific CSV file.

    Args:
        states (list): List of state names.
        file_name (str): CSV file name in each state's folder.
        stat_name (str): Target name from 'Name' column.
        method (str): Aggregation method: 'sum', 'average', or 'median'.

    Returns:
        float: Aggregated result (or None if no values found).
    """
    values = []

    for state in states:
        file_path = os.path.join(base_path, state, file_name)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["Name"] = df["Name"].astype(str).str.strip()
                row = df[df["Name"] == stat_name]
                if not row.empty:
                    val = str(row["Value"].values[0]).replace(",", "")
                    if val:
                        values.append(float(val))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

    if not values:
        return None

    if method == "sum":
        return sum(values)
    elif method == "average":
        return sum(values) / len(values)
    elif method == "median":
        return sorted(values)[len(values) // 2]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")



# --- Informations for Frost States

# Metrics
metrics_by_category = {
    "Demographic Characteristics": {
        "file": "Demographic characteristics.csv",
        "metrics": [
            ("Total Population", "Total population", "sum", "units", True)
        ]
    },
    "Socioeconomic Characteristics": {
        "file": "Socioeconomic characteristics.csv",
        "metrics": [
            ("Average Income per Year", "Average income", "average", "USD", True),
            ("Median Household Income per Year", "Median household income", "median", "USD", True),
            ("Average Household Size", "Average household size", "average", "units_1dec", True),
            ("Percent speaking Spanish at Home", "Percent speaking Spanish at home", "average", "percent", True),
            ("Percent speaking other Indo-European Languages at Home", "Percent speaking other Indo-European languages at home", "average", "percent", True)
        ]
    },
    "Housing Characteristics": {
        "file": "Housing characteristics.csv",
        "metrics": [
            ("Total Housing Units", "Total number of housing units", "sum", "units", True),
            ("Average Housing Value", "Average housing value", "average", "USD", True),
            ("Median Year Structure Built", "Median year structure built", "median", "year", True),
            ("Median Monthly Owner Costs", "Median selected monthly owner cost", "average", "USD", True),
            ("Vacancy Rate", "Vacancy rate", "average", "percent", True)
        ]
    },
    "Building Permits": {
        "file": "Building Permits.csv",
        "metrics": [
            ("1 Unit Buildings", "1 Unit Buildings", "sum", "units", True),
            ("1 Unit Building Value ($1,000)", "1 Unit Building Value ($1,000)", "sum", "USD", True),
            ("2 Unit Buildings", "2 Unit Buildings", "sum", "units", True),
            ("2 Unit Building Value ($1,000)", "2 Unit Building Value ($1,000)", "sum", "USD", True),
            ("3-4 Unit Buildings", "3-4 Unit Buildings", "sum", "units", False),
            ("3-4 Unit Building Value ($1,000)", "3-4 Unit Building Value ($1,000)", "sum", "USD", False),
            ("5+ Unit Buildings", "5+ Unit Buildings", "sum", "units", False),
            ("5+ Unit Building Value ($1,000)", "5+ Unit Building Value ($1,000)", "sum", "USD", False)
        ]
    },
    "Consumer Expenditures": {
        "file": "Consumer Spending.csv",
        "metrics": [
            ("Average Consumer Expenditures per Household", "Total consumer expenditures per household", "average", "USD", True),
            ("Average Consumer Expenditures per Household on Housing", "Consumer expenditures per household on Housing", "average", "USD", True),
            ("Average Consumer Expenditures per Household on Home Improvements", "Consumer expenditures per household on Home improvements", "average", "USD", True),
            ("Average Consumer Expenditures per Household on Water", "Consumer expenditures per household on Water", "average", "USD", True),
        ]
    },
}

try: 
    # Metrics bestimmen
    for category_label, config in metrics_by_category.items():
        csv_file = config["file"]
        metric_list = config["metrics"]

        # Aggregierte Werte berechnen
        values = [
            get_aggregated_stat_from_states(
                selected_states,
                csv_file,
                metric_name,
                method
            )
            for _, metric_name, method, _, show in metric_list if show
        ]

        # Section-Überschrift
        st.subheader(f"{category_label}")

        visible_metrics = [m for m in metric_list if m[4]]
        cols = st.columns(len(visible_metrics))

        # Dynamische Spaltenanzeige
        for col, (label, _, _, unit, _), value in zip(cols, visible_metrics, values):
            if value is None:
                col.metric(label, "N/A")
            else:
                if unit == "%" or unit == "percent":
                    formatted = f"{value:.1f} %"
                elif unit == "USD":
                    formatted = f"$ {value:,.0f}"
                elif unit == "year":
                    formatted = f"{int(value)}"
                elif unit == "units":
                    formatted = f"{value:,.0f}"
                elif unit == "units_1dec":
                    formatted = f"{value:,.1f}"
                elif unit == "units_2dec":
                    formatted = f"{value:,.2f}"
                else:
                    formatted = f"{value:,.1f} {unit}"

                col.metric(label, formatted)


    # --- data by states
    def build_data_by_states_dataframe(states, metrics_by_category):
        """
        Erstellt ein DataFrame mit allen Metriken für alle ausgewählten Staaten.

        Returns:
            pd.DataFrame: Zeilen = Metriken, Spalten = Bundesstaaten
        """
        data = {}

        for state in states:
            state_values = {}
            for category_config in metrics_by_category.values():
                file = category_config["file"]
                for label, metric_name, _, _, _ in category_config["metrics"]:
                    file_path = os.path.join(base_path, state, file)
                    if os.path.exists(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            df["Name"] = df["Name"].astype(str).str.strip()
                            row = df[df["Name"] == metric_name]
                            if not row.empty:
                                val = str(row["Value"].values[0]).replace(",", "")
                                if val:
                                    state_values[label] = float(val)
                        except Exception as e:
                            print(f"Fehler in {state} – {label}: {e}")
            data[state] = state_values

        df_out = pd.DataFrame(data)
        df_out = df_out.T
        return df_out



    # --- Data download
    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Data by States')
        return output.getvalue()

    data_by_states = build_data_by_states_dataframe(selected_states, metrics_by_category)

    # Excel-Daten generieren
    excel_bytes = convert_df_to_excel(data_by_states)

    # Download-Button im Dashboard anzeigen
    st.download_button(
        label="Download data as Excel",
        data=excel_bytes,
        file_name="data_by_states.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



    # --- data for a metric for states
    # Alle Metriken in ein Dictionary extrahieren
    metric_options = []
    metric_lookup = {}  # Label → (file, stat_name, method)

    for category_config in metrics_by_category.values():
        file = category_config["file"]
        for label, stat_name, method, _, show_column in category_config["metrics"]:
            if show_column:
                metric_options.append(label)
                metric_lookup[label] = (file, stat_name, method)

    def data_states(metric, states):
        states_data = []
        for state in states:
            file, stat_name, method = metric_lookup[metric]
            value = get_aggregated_stat_from_states([state], file, stat_name, method)
            states_data.append({
                "state": state,
                "abbr": state_abbr[state],  # ← wichtig!
                "value": value
            })
        return states_data



    # --- Boxplots

    # Metriken für die Boxplots
    box_metrics = ["Average Housing Value", "Total Housing Units", "Median Year Structure Built"]

    # Daten vorbereiten
    df_boxes = pd.DataFrame()
    for metric in box_metrics:
        data = data_states(metric, selected_states)
        df = pd.DataFrame(data)
        df["metric"] = metric
        df_boxes = pd.concat([df_boxes, df], ignore_index=True)

    # Subplots vorbereiten
    fig_multi = make_subplots(
        rows=1, cols=len(box_metrics),
        subplot_titles=box_metrics,
        shared_yaxes=False
    )

    # Pro Metrik ein Boxplot erzeugen
    for i, metric in enumerate(box_metrics, start=1):
        df_metric = df_boxes[df_boxes["metric"] == metric]
        fig_multi.add_trace(
            go.Box(
                y=df_metric["value"],
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                name=metric,
                marker_color="steelblue",
                hovertext=df_metric["state"],
                hoverinfo="text+y"
            ),
            row=1, col=i
        )

    # Layout anpassen
    fig_multi.update_layout(
        font=common_font,
        height=700,
        showlegend=False)

    for i in range(1, len(box_metrics) + 1):
        fig_multi.update_xaxes(tickfont=common_font, title_font=common_font, row=1, col=i)
        fig_multi.update_yaxes(tickfont=common_font, title_font=common_font, row=1, col=i)


    st.subheader(f"Distribution of selected Metrics across Selected States")
    st.plotly_chart(fig_multi, use_container_width=True)



    # --- Boxplot individuell
    col1, col2, col3 = st.columns([1, 1, 1])  # Verhältnis der Spaltenbreiten
    with col2:
        selected_box_metric = st.selectbox("", metric_options)

    df_box_go = pd.DataFrame(data_states(selected_box_metric, selected_states))
    df_box_go = df_box_go.dropna(subset=["value"])

    fig_go_box = go.Figure()

    fig_go_box.add_trace(go.Box(
        y=df_box_go["value"],
        boxpoints="all",
        jitter=0.4,  # Punkte leicht verteilen
        pointpos=0,  # Punkte zentriert auf Box
        marker_color="steelblue",
        name=selected_box_metric,
        hovertext=df_box_go["state"],
        hoverinfo="text+y"))



    fig_go_box.update_layout(
        title=f"Distribution of {selected_box_metric} across Selected States",
        yaxis_title=selected_box_metric,
        font=common_font,
        height=600,
        showlegend=False)

    fig_go_box.update_layout(
        font=common_font,
        #title_font=common_font,
        xaxis_title_font=common_font,
        yaxis_title_font=common_font,
        xaxis=dict(tickfont=common_font),
        yaxis=dict(tickfont=common_font))

    col1, col2, col3 = st.columns([1, 1, 1])  # Verhältnis der Spaltenbreiten

    with col2:
        st.plotly_chart(fig_go_box, use_container_width=True)



    # --- load Table with data by state
    def aggregate_named_table_by_state(states, filename, method="average"):
        """
        Aggregiert alle 'Name'–'Value'-Paare aus einer CSV über mehrere Staaten hinweg.

        Args:
            states (list): Liste von Bundesstaaten
            filename (str): CSV-Dateiname
            method (str): Aggregationsmethode: 'sum' oder 'average'

        Returns:
            pd.DataFrame mit Spalten 'Name', 'Value'
        """
        dfs = []
        for state in states:
            path = os.path.join(base_path, state, filename)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    dfs.append(df[["Name", "Value"]])
                except Exception as e:
                    print(f"Fehler bei {state}: {e}")
        if not dfs:
            return None
        df_all = pd.concat(dfs)
        df_all["Name"] = df_all["Name"].str.replace("Consumer expenditures per household on ", "", regex=False).str.strip()

        if method == "average":
            df_agg = df_all.groupby("Name", as_index=False, sort=False)["Value"].mean()
        elif method == "sum":
            df_agg = df_all.groupby("Name", as_index=False, sort=False)["Value"].sum()
        else:
            raise ValueError("Unknown method: use 'sum' or 'average'.")
        
        df_agg["Reihenfolge"] = range(len(df_agg))    
        return df_agg


    main_categories = [
        "Alcoholic beverages", "Apparel & services", "Dining out (Food away from home)", "Education", 
        "Entertainment / Recreation", "Food consumed at home", "Health Care", "Household Services",
        "Housing", "Life and other insurance", "Pensions and social security", "Personal Care Products and Services", 
        "Local Transportation", "Travel", "Child Care", "Day Care/Nursery & Preschools", "Babysitting/Child Care in Own/Other Home"
    ]

    df_consumer_expenditures = aggregate_named_table_by_state(selected_states, "Consumer Spending.csv", method="average")

    df_consumer_expenditures["Name"] = df_consumer_expenditures["Name"].str.strip()
    df_consumer_expenditures = df_consumer_expenditures[df_consumer_expenditures["Name"] != "Total consumer expenditures per household"]

    # Kategoriezuweisung & Formatierung
    df_consumer_expenditures["Kategorie_Art"] = df_consumer_expenditures["Name"].apply(lambda x: "Oberkategorie" if x in main_categories else "Detail")
    df_consumer_expenditures["Reihenfolge"] = range(len(df_consumer_expenditures))
    df_consumer_expenditures = df_consumer_expenditures.sort_values(by="Reihenfolge")
    df_consumer_expenditures["Name_fett"] = df_consumer_expenditures.apply(lambda row: f"<b>{row['Name']}</b>" if row["Kategorie_Art"] == "Oberkategorie" else row["Name"],axis=1)

    # Balkendiagramm
    fig = px.bar(
        df_consumer_expenditures,
        x="Value",
        y="Name_fett",
        labels={"Value": "Value [US-$/year]", "Name_fett": "Consumer Expenditure"},
        orientation="h",
        color="Kategorie_Art",
        color_discrete_map={"Oberkategorie": "darkblue", "Detail": "lightblue"}
    )

    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=df_consumer_expenditures["Name_fett"].tolist()[::-1]
        ),
        showlegend=False,
        height=20 * len(df_consumer_expenditures)
    )

    fig.update_layout(
        font=common_font,
        #title_font=common_font,
        xaxis_title_font=common_font,
        yaxis_title_font=common_font,
        xaxis=dict(tickfont=common_font),
        yaxis=dict(tickfont=common_font))

    # Diagramm anzeigen
    st.subheader(f"Overview of Consumer Expenditures in Selected States")
    st.plotly_chart(fig, use_container_width=True)



    # --- Heatmap USA (unabhängig von Auswahl)
    st.subheader("Heatmap USA")

    # Dropdown zur Auswahl der Metrik
    selected_metric_label = st.selectbox("Select a metric for the heatmap", metric_options)

    df_heatmap = pd.DataFrame(data_states(selected_metric_label, all_states))

    # Karte zeichnen
    fig_heatmap = px.choropleth(
        df_heatmap,
        locations="abbr",
        locationmode="USA-states",
        color="value",
        scope="usa",
        hover_name="state",
        hover_data={"abbr": False, "value": True},
        color_continuous_scale="Turbo",
        labels={"value": ""}
    )


    fig_heatmap.update_layout(
        geo=dict(showlakes=False, lakecolor="lightblue"),
        dragmode=False,
        uirevision="static",
        coloraxis_showscale=True,
        margin=dict(l=0, r=0, t=40, b=0),
        height=500
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)



    # --- Textfeld
    st.markdown("<p style='font-size: 0.8rem; color: gray;'>"
                "<br>"
                "Sources: <br>" 
                "<br>"
                "U.S. Census Bureau, available online at: <a href='https://cbb.census.gov/' target='_blank'>https://cbb.census.gov/</a>.<br>"
                "<br>"
                "Demographic Characteristics: 2023 American Community Survey 5-year Data Profile.<br>"
                "Socioeconomic Characteristics: 2023 American Community Survey 5-year Data Profile.<br>"
                "Housing Characteristics: 2023 American Community Survey 5-year Data Profile.<br>"
                "Building Permits: 2023 Building Permits Survey.<br>"
                "Consumer Spending: 2024 Esri Consumer Spending data."
                "</p>", unsafe_allow_html=True)

except KeyError:
    st.write("")
