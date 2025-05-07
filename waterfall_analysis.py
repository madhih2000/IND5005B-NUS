import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import os
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def generate_weeks_range(start_week, num_weeks=12):
    #weeks_range = [f"WW{str((start_week + i - num_weeks) % 52 or 52).zfill(2)}" for i in range(2 * num_weeks + 1)]
    weeks_range = []
    for i in range(2 * num_weeks + 1):
        week = (start_week + i - num_weeks) % 52 or 52
        if start_week-num_weeks < 1 and week > start_week + num_weeks: #edge case in case you want to check shortage data for week 2 ,e.g.
            continue
        weeks_range.append(f"WW{str(week).zfill(2)}")
    print(weeks_range)
    return weeks_range

# Function to normalize column names
def normalize(col):
    return col.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').lower()

def save_to_excel(df, output_file):
    """Saves a DataFrame to an Excel file."""
    if df is not None:
        try:
            df.to_excel(output_file, index=False)
            print(f"Data saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
    else:
        print("No data to save.")

def extract_and_aggregate_weekly_data(folder_path, material_number, plant, site, start_week, num_weeks=12):
    """
    Extracts and aggregates weekly data for a specific material number, plant, and site,
    starting from a specified week and including the next 'num_weeks' weeks.

    Args:
        folder_path (str): Path to the folder containing the XLSX files.
        material_number (str): Material number to filter.
        plant (str): Plant to filter.
        site (str): Site to filter.
        start_week (str): Starting week (e.g., "WW32").
        num_weeks (int): Number of weeks to include in the output.

    Returns:
        pandas.DataFrame: DataFrame containing the aggregated weekly data, or None if no data is found.
    """

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return None
    
    print(folder_path)

    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".xlsx")])

    if not all_files:
        print(f"Error: No XLSX files found in '{folder_path}'.")
        return None

    week_numbers = list(range(max(1, start_week - num_weeks), start_week + 1))
    #print(week_numbers)

    selected_data = []
    selected_weeks = []

    #Columns to add to the final df
    initial_columns = ["MaterialNumber", "Plant", "Site", "Measures", "InventoryOn-Hand", "LeadTime(Week)"]
    weeks_range = generate_weeks_range(start_week,num_weeks)
    #print(weeks_range)

    for i in week_numbers:
        week_file = f"WW{i}.xlsx"
        week_col = f"WW{i:02d}"

        if week_file in all_files:
            file_path = os.path.join(folder_path, week_file)
            try:
                df = pd.read_excel(file_path)
                filtered_df = df[
                    (df["Material Number"] == material_number)
                    & (df["Plant"] == plant)
                    & (df["Site"] == site)
                ]

                lead_value = 0

                if i == start_week:

                    # Lead Time column, normalized
                    target_col = normalize("Lead Time (Week)")
                    lead_col = None

                    # Loop to find the unnormalized matching column
                    for col in df.columns:
                        if normalize(col) == target_col:
                            lead_col = col
                            break

                    if lead_col:
                        lead_value = df.loc[df["Material Number"] == material_number, lead_col]

                
                #Remove whitespace in column names
                filtered_df.columns = filtered_df.columns.str.strip().str.replace('\n', ' ', regex=False).str.replace(r'\s+', '', regex=True)
        
                if not filtered_df.empty:
                    # Select only the required columns
                    all_columns = initial_columns + weeks_range

                    # Check if all columns exist in filtered_df
                    missing_columns = [col for col in all_columns if col not in filtered_df.columns]
                    
                    if missing_columns:
                        raise ValueError(f"Missing columns in {week_file}: {', '.join(missing_columns)}")
                        return None
                    
                    filtered_df = filtered_df[all_columns]

                    #Add new column snapshot
                    filtered_df['Snapshot'] = week_col

                    # Remove the first element from weeks_range for next iteration
                    weeks_range = weeks_range[1:]

                    selected_data.append(filtered_df)
                    selected_weeks.append(f"WW{i:02d}")

                else:
                    # If filtered_df is empty, create a new dataframe with NaN values
                    all_columns = initial_columns + weeks_range
                    empty_df = pd.DataFrame(columns=all_columns)
                    empty_df['Snapshot'] = week_col  # Add the Snapshot column with the week_col value

                    # Add the empty dataframe to selected_data and weeks
                    selected_data.append(empty_df)
                    selected_weeks.append(f"WW{i:02d}")

            except FileNotFoundError:
                print(f"Error: File '{week_file}' not found.")
            except Exception as e:
                print(f"Error reading file '{week_file}': {e}")
        else:
            print(f"Warning: Week file '{week_file}' not found.")

    if not selected_data:
        print("No matching data found.")
        return None

    result_df = pd.concat(selected_data, ignore_index=True)
    
    #reorder columns to have week at the beginning.
    cols = result_df.columns.tolist()
    cols = ['Snapshot'] + [col for col in cols if col != 'Snapshot']
    result_df = result_df[cols]

    return result_df, lead_value

def plot_stock_prediction_plotly(df, start_week, lead_time, weeks_range):
    """Plots actual and predicted stock values over weeks."""

    weeks = generate_weeks_range(start_week, weeks_range)
    stock_values = df[df['Measures'] == 'Weeks of Stock']

    fig = go.Figure()

    actual_values = []
    predicted_values = []
    plot_weeks_actual = []
    plot_weeks_predicted = []

    for i, week in enumerate(weeks):
        print(i)
        print(predicted_values)
        if week not in df.columns:
            print(f"Warning: Week {week} not found in DataFrame.")
            continue  # skip the iteration if the week is not in the dataframe.

        if week in df['Snapshot'].values: # Only get actual values if the week is in the snapshot.
            actual_value = stock_values[stock_values['Snapshot'] == week][week].iloc[0]
            if actual_value is None or np.isnan(actual_value):
                actual_value = 0  # if actual value is none or nan, set to 0.
            actual_values.append(actual_value)
            plot_weeks_actual.append(week)
        else:
            break

        if i != 0:
            predicted_value = stock_values[stock_values['Snapshot'] == weeks[i - 1]][week].iloc[0]
        else:
            predicted_value = None

        if predicted_value is None or np.isnan(predicted_value):
            if i > 0:
                predicted_value = predicted_values[-1]  # use the previous predicted value if the current one is None or NaN
            else:
                predicted_value = 0  # skip the first one as it is none.

        predicted_values.append(predicted_value)
        plot_weeks_predicted.append(week)

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Week': plot_weeks_actual,
        'Actual': actual_values,
        'Predicted': predicted_values[:len(plot_weeks_actual)]  # Align lengths
    })

    # Add Deviation Flag
    def check_deviation_detail(row):
        if row['Actual'] == 0 and row['Predicted'] == 0:
            return False, ""
        elif row['Actual'] == 0 and row['Predicted'] != 0:
            return True, "Actual demand is 0, but forecasted is not"
        elif row['Actual'] != 0 and row['Predicted'] == 0:
            return True, "Forecasted demand expected to be 0, but actual is not"
        ratio = row['Predicted'] / row['Actual']
        if ratio >= 1.5:
            percent = round((ratio - 1) * 100, 1)
            return True, f"Overestimate by {percent}%"
        elif ratio <= (1 / 1.5):
            percent = round((1 - ratio) * 100, 1)
            return True, f"Underestimate by {percent}%"
        else:
            return False, ""

    deviation_results = comparison.apply(check_deviation_detail, axis=1, result_type='expand')
    comparison['Deviation_Flag'] = deviation_results[0]
    comparison['Deviation_Detail'] = deviation_results[1]

    fig.add_trace(go.Scatter(x=plot_weeks_actual, y=actual_values, mode='lines+markers', name='Actual Weeks of Stock'))
    fig.add_trace(go.Scatter(x=plot_weeks_predicted, y=predicted_values, mode='lines+markers', name='Predicted Weeks of Stock'))

    fig.update_layout(title='Actual vs. Predicted Weeks of Stock',
                      xaxis_title='Week',
                      yaxis_title='Weeks of Stock')

    return actual_values, fig, comparison

def identify_specific_po_timing_issues(demand_df, po_df):
    """
    Scenario 2: Match POs to actual weekly demand using Inventory-On-Hand
    and provide actionable push/pull suggestions.

    Args:
        demand_df: DataFrame containing waterfall data with 'Snapshot', 'Measures', and WW columns.
        po_df: DataFrame with ['Purchasing Document', 'Order WW', 'GR WW', 'GR Quantity'].

    Returns:
        A DataFrame showing demand coverage, PO matching, unmet demand, and suggested PO timing actions.
    """
    demand_rows = demand_df[demand_df['Measures'] == 'Demand w/o Buffer']
    supply_rows = demand_df[demand_df['Measures'] == 'Supply']  # corrected to use Supply for inventory
    weeks = sorted([col for col in demand_df.columns if col.startswith("WW")])
    snapshots = demand_rows['Snapshot'].unique()

    po_df = po_df.copy()
    po_df['Remaining Qty'] = po_df['GR Quantity']

    results = []

    for snapshot in snapshots:
        week = snapshot  # e.g., WW08
        week_num = int(week.replace("WW", ""))

        # Get demand for this actual week
        demand_val = demand_rows[demand_rows['Snapshot'] == snapshot][week]
        demand = int(demand_val.values[0]) if not demand_val.empty and not pd.isna(demand_val.values[0]) else 0

        # ✅ Corrected: Get Inventory-On-Hand from the 'Supply' row using the week column
        inv_val = supply_rows[supply_rows['Snapshot'] == snapshot]['InventoryOn-Hand']
        inventory_on_hand = int(inv_val.values[0]) if not inv_val.empty and not pd.isna(inv_val.values[0]) else 0

        matched_pos = []
        remaining_demand = demand

        # Use inventory first
        inventory_used = min(inventory_on_hand, remaining_demand)
        remaining_demand -= inventory_used

        # Use POs arriving in same week
        available_pos = po_df[(po_df['GR WW'] == week_num) & (po_df['Remaining Qty'] > 0)]
        for _, po in available_pos.iterrows():
            po_id = po['Purchasing Document']
            qty = po['Remaining Qty']
            used = min(qty, remaining_demand)

            po_df.loc[po_df['Purchasing Document'] == po_id, 'Remaining Qty'] -= used
            matched_pos.append(f"{po_id} (Used {used})")
            remaining_demand -= used

            if remaining_demand <= 0:
                break

        # Determine if POs need adjustment
        flags = []

        if remaining_demand > 0:
            # Pull in future POs
            late_pos = po_df[(po_df['GR WW'] > week_num) & (po_df['Remaining Qty'] > 0)]
            for _, po in late_pos.iterrows():
                flags.append(f"Pull In: PO {po['Purchasing Document']} (ETA WW{po['GR WW']}, Qty {po['Remaining Qty']})")

        # Check if there are early POs not needed yet
        early_pos = po_df[(po_df['GR WW'] < week_num) & (po_df['Remaining Qty'] > 0)]
        for _, po in early_pos.iterrows():
            flags.append(f"Push Out: PO {po['Purchasing Document']} (ETA WW{po['GR WW']}, Qty {po['Remaining Qty']})")

        results.append({
            "Snapshot Week": snapshot,
            "Week": week,
            "Demand (Actual)": demand,
            "Inventory-On-Hand": inventory_on_hand,
            "Inventory Used": inventory_used,
            "Matched PO Lines": ", ".join(matched_pos) if matched_pos else "None",
            "Unmet Demand": max(0, remaining_demand),
            "Actions Needed": "; ".join(flags) if flags else "OK"
        })

    return pd.DataFrame(results)

def check_wos_against_lead_time(wos_list, lead_time):
    """
    Compares values in wos_list against lead_time and flags close matches.

    Args:
        wos_list: A list of floats, potentially containing 0.
        lead_time: An integer representing the target lead time.

    Returns:
        A list of messages indicating the comparison results and a boolean indicating
        whether an immediate order is needed.
    """
    # Check if lead_time is a pandas Series and handle it
    if isinstance(lead_time, pd.Series):
        lead_time_value = lead_time.iloc[0]
    else:
        lead_time_value = lead_time  # If it's already a scalar


    messages = []
    order_immediately = False
    tolerance = 0.1  # 10% tolerance

    for i, wos in enumerate(wos_list):
        if wos == 0:
            #messages.append(f"Index {i}: No value recorded.")
            continue

        lower_bound = lead_time_value * (1 - tolerance)
        upper_bound = lead_time_value * (1 + tolerance)

        if wos < lower_bound:
            messages.append(f"Index {i}: Value {wos} is significantly lower than lead time {lead_time_value}. ALERT!")
        elif wos > upper_bound:
            continue
            #messages.append(f"Index {i}: Value {wos} is slightly higher than lead time {lead_time_value}. Warning.")

    if wos_list and wos_list[-1] != 0:
        if wos_list[-1] < lead_time_value*0.5: # if the last value is less than half the lead time.
            order_immediately = True
            messages.append("Last recorded value is significantly low, consider immediate order.")
        else:
            messages.append("Last recorded value does not indicate immediate order is needed.")

    return messages, order_immediately

def apply_coloring_to_output(excel_buffer, lead_time):
    # Rewind buffer and load workbook
    excel_buffer.seek(0)
    wb = load_workbook(excel_buffer)
    ws = wb['Sheet1']

    # Define colors
    red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    green_fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")

    # Identify relevant columns
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    measures_col_idx = header.index('Measures') + 1
    ww_col_indices = [i + 1 for i, h in enumerate(header) if str(h).startswith("WW")]

    # Apply coloring logic
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        if row[measures_col_idx - 1].value == 'Weeks of Stock':
            for idx in ww_col_indices:
                cell = row[idx - 1]
                raw_val = cell.value
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    continue  # Skip non-numeric

                
                if val < 0:
                    cell.fill = red_fill
                elif val < lead_time.iloc[0]:
                    cell.fill = yellow_fill
                else:
                    cell.fill = green_fill

    # Save again to a new BytesIO
    colored_output = BytesIO()
    wb.save(colored_output)
    colored_output.seek(0)
    return colored_output

#Condition 4: Longer delivery lead time
def lead_time_check(result_df):
    # First, group by Snapshot and collect the unique LeadTime(Week) values per snapshot
    leadtime_changes = ( 
    result_df.groupby("Snapshot")["LeadTime(Week)"]
    .unique()
    .reset_index()
    )

    # function to check for changes and generate a string of changes
    def detect_changes(row):
        leadtimes = row["LeadTime(Week)"]
        if len(leadtimes) > 1:
            change_str = f"Lead time changes detected: {', '.join(map(str, leadtimes))}"
            return True, change_str
        else:
            return False, "No change"

    # Apply the function to create new columns
    leadtime_changes[["Changed", "Change Details"]] = leadtime_changes.apply(
        detect_changes, axis=1, result_type="expand"
    )

    return leadtime_changes

#Condition 6: Immediate demand increase within lead time
def check_demand(df):
    # Prepare for analysis: Convert snapshots and WW column names to align
    ww_columns = [col for col in df.columns if col.startswith("WW")]
    snapshots = df["Snapshot"].unique()

    #Algo check
    standout_weeks_info = []

    for snapshot in snapshots:
        # Filter for current snapshot and specifically the "Demand w/o Buffer"
        snapshot_df = df[(df["Snapshot"] == snapshot) & (df["Measures"] == "Demand w/o Buffer")]

        if snapshot_df.empty:
            continue  # Skip if no relevant data
        
        # Extract lead time and demand value from the respective WW column
        lead_time = int(snapshot_df["LeadTime(Week)"].values[0])
        ww_value = snapshot_df[snapshot].values[0]

        if pd.isna(ww_value):
            continue  # Skip if demand data for snapshot week is NaN

        # Determine index of snapshot week (e.g., WW07 → index in ww_columns list)
        try:
            current_index = ww_columns.index(snapshot)
        except ValueError:
            continue  # Skip if snapshot doesn't match WW column

        # Calculate the forward-looking range within bounds
        end_index = min(current_index + lead_time, len(ww_columns))
        future_weeks = ww_columns[current_index+1:end_index+1]

        # Extract future demand values
        future_values = snapshot_df[future_weeks].values.flatten()

        # Compare each future week's demand with current snapshot week's value
        for week, future_demand in zip(future_weeks, future_values):
            if pd.notna(future_demand) and ww_value > 0 and future_demand / ww_value >= 2:
                standout_weeks_info.append({
                    "Snapshot": snapshot,
                    "Current_Week": snapshot,
                    "LeadTime": lead_time,
                    "BaseDemand": ww_value,
                    "SpikeWeek": week,
                    "SpikeDemand": future_demand,
                    "Multiplier": round(future_demand / ww_value, 2)
                })

    final_pd = pd.DataFrame(standout_weeks_info)
    if final_pd.empty:
        final_msg = "There is no immediate demand increase within lead time of the material number."
    else:
        final_msg = "There are instances of immediate demand increase within lead time of the material number."

    # Display explanation
    st.info(f"A spike is flagged if the demand in any week within the {lead_time}-week lead time exceeds twice the demand of the current week.")

    return  final_pd, final_msg

# Scenario 5: Identifying irregular consumption patterns
def analyze_week_to_week_demand_changes(result_df, abs_threshold=10, pct_threshold=0.3):
    """
    Filters for 'Demand w/o Buffer', calculates demand values based on snapshot columns,
    and detects week-to-week anomalies such as spikes or drops.

    Returns:
        - output_df: DataFrame with demand values and irregular pattern flags
    """
    # Step 1: Filter
    filtered_df = result_df[result_df['Measures'] == 'Demand w/o Buffer'].copy()
    if filtered_df.empty:
        raise ValueError("No rows found with Measures == 'Demand w/o Buffer'")

    output_rows = []

    for index, row in filtered_df.iterrows():
        snapshot_col = row["Snapshot"]
        if snapshot_col in filtered_df.columns:
            demand_value = row[snapshot_col]
            output_rows.append({
                "Snapshot": snapshot_col,
                "Material Number": row["MaterialNumber"],
                "Plant": row["Plant"],
                "Site": row["Site"],
                "Demand w/o Buffer": demand_value
            })

    # Step 2: Create base output DataFrame
    output_df = pd.DataFrame(output_rows)

    # Step 3: Sort and calculate week-over-week changes
    output_df = output_df.sort_values(by="Snapshot").reset_index(drop=True)
    output_df["WoW Change"] = output_df["Demand w/o Buffer"].diff()
    output_df["WoW % Change"] = output_df["Demand w/o Buffer"].pct_change()

    # Step 4: Flag irregularities
    output_df["Spike"] = output_df["WoW Change"] > abs_threshold
    output_df["Drop"] = output_df["WoW Change"] < -abs_threshold
    output_df["Sudden % Spike"] = output_df["WoW % Change"] > pct_threshold
    output_df["Sudden % Drop"] = output_df["WoW % Change"] < -pct_threshold

    # Step 5: Display explanation
    st.info(f"A spike or drop is flagged if the week-over-week change exceeds ±{abs_threshold} units "
            f"or ±{int(pct_threshold * 100)}%.")

    return output_df


def scenario_1(waterfall_df, po_df):
    # Filter relevant rows
    supply_rows = waterfall_df[waterfall_df['Measures'] == 'Supply']
    demand_rows = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer']

    # Get initial snapshot and inventory from Supply rows
    initial_snapshot = supply_rows['Snapshot'].iloc[0]
    initial_inventory_calc = int(supply_rows[supply_rows['Snapshot'] == initial_snapshot]['InventoryOn-Hand'].values[0])

    # All snapshots to iterate through
    snapshots = waterfall_df['Snapshot'].unique()

    results = []
    current_inventory_calc = initial_inventory_calc

    for snapshot in snapshots:
        week_col = snapshot
        week_num = int(snapshot.replace("WW", ""))

        # Get supply and demand values
        demand_val = demand_rows[demand_rows['Snapshot'] == snapshot][week_col]
        supply_val = supply_rows[supply_rows['Snapshot'] == snapshot][week_col]

        demand = int(demand_val.values[0]) if not demand_val.empty else 0
        supply = int(supply_val.values[0]) if not supply_val.empty else 0

        # Start inventory from Waterfall column
        inv_val = supply_rows[supply_rows['Snapshot'] == snapshot]['InventoryOn-Hand']
        start_inventory_waterfall = int(inv_val.values[0]) if not inv_val.empty else 0

        # GR quantity from PO data
        po_received = 0
        if not po_df.empty:
            po_received = po_df[po_df['GR WW'] == week_num]['GR Quantity'].sum()

        # Calculate end inventories INCLUDING PO received
        end_inventory_calc = current_inventory_calc + supply + po_received - demand
        end_inventory_waterfall = start_inventory_waterfall + supply + po_received - demand

        # Flag logic
        flags = []
        if po_df.empty:
            if supply > 0:
                flags.append("No PO data available to validate supply")
        else:
            if po_received < supply:
                if po_received == 0 and supply > 0:
                    flags.append("No PO received for expected supply")
                elif 0 < po_received < supply:
                    flags.append(f"Partial PO received: Expected {supply}, Got {po_received}")
                else:
                    flags.append("Supply in Waterfall not backed by PO receipts")

        if end_inventory_calc < 0:
            flags.append("Inventory went negative — demand exceeded supply and stock")

        # Record results
        results.append({
            'Snapshot Week': snapshot,
            'Start Inventory (Waterfall)': start_inventory_waterfall,
            'Start Inventory (Calc)': current_inventory_calc,
            'Demand (Waterfall)': demand,
            'Supply (Waterfall)': supply,
            'PO GR Quantity': po_received,
            'End Inventory (Waterfall)': end_inventory_waterfall,
            'End Inventory (Calc)': end_inventory_calc,
            'Flags': ", ".join(flags) if flags else "OK"
        })

        # Update for next loop
        current_inventory_calc = end_inventory_calc

    return pd.DataFrame(results)

def scenario_2(waterfall_df, po_df):
    # Filter relevant rows
    supply_rows = waterfall_df[waterfall_df['Measures'] == 'Supply']
    demand_rows = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer']

    # Initial snapshot and inventory
    initial_snapshot = supply_rows['Snapshot'].iloc[0]
    initial_inventory_calc = int(supply_rows[supply_rows['Snapshot'] == initial_snapshot]['InventoryOn-Hand'].values[0])

    snapshots = waterfall_df['Snapshot'].unique()
    results = []
    current_inventory_calc = initial_inventory_calc
    current_inventory_sim = initial_inventory_calc

    used_po_docs = set()  # Track POs used in simulation

    for i, snapshot in enumerate(snapshots):
        week_col = snapshot
        week_num = int(snapshot.replace("WW", ""))

        # Supply & demand
        demand_val = demand_rows[demand_rows['Snapshot'] == snapshot][week_col]
        supply_val = supply_rows[supply_rows['Snapshot'] == snapshot][week_col]
        demand = int(demand_val.values[0]) if not demand_val.empty else 0
        supply = int(supply_val.values[0]) if not supply_val.empty else 0

        # Inventory from Waterfall
        inv_val = supply_rows[supply_rows['Snapshot'] == snapshot]['InventoryOn-Hand']
        start_inventory_waterfall = int(inv_val.values[0]) if not inv_val.empty else 0

        # GR quantity and PO documents (actual)
        po_received_data = po_df[(po_df['GR WW'] == week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))]
        po_received = po_received_data['GR Quantity'].sum()
        po_docs_received = list(po_received_data['Purchasing Document'])

        # Base calculation
        end_inventory_calc = current_inventory_calc + supply + po_received - demand
        end_inventory_waterfall = start_inventory_waterfall + supply + po_received - demand

        # Root cause flag logic
        flags = []
        action = "No Action"
        suggested_po = None
        adjusted_gr_week = None
        simulated_po_qty = 0

        # ---------------- SIMULATION ----------------

        # Start with the same inventory in simulation
        simulated_supply = supply
        simulated_po_received = po_received
        simulated_end_inventory = current_inventory_sim + simulated_supply + simulated_po_received - demand

        # Simulate pull-in if inventory went negative
        if simulated_end_inventory < 0:
            flags.append("Inventory went negative — stockout")
            future_po = po_df[
                (po_df['GR WW'] > week_num) &
                (~po_df['Purchasing Document'].isin(used_po_docs))
            ]
            if not future_po.empty:
                pulled_po = future_po.iloc[0]
                suggested_po = pulled_po['Purchasing Document']
                simulated_po_qty = pulled_po['GR Quantity']
                adjusted_gr_week = week_num  # Pull in to current week
                simulated_po_received += simulated_po_qty
                simulated_end_inventory = current_inventory_sim + simulated_supply + simulated_po_received - demand
                action = "Pull In"
                used_po_docs.add(suggested_po)

        # Simulate push-out if excess PO not needed
        elif simulated_end_inventory > current_inventory_sim + simulated_supply - demand and po_received > 0:
            flags.append("Inventory built up unnecessarily — potential overstock")
            suggested_po = po_docs_received[0] if po_docs_received else None
            adjusted_gr_week = week_num + 1  # Hypothetical push to next week
            simulated_po_received = 0  # Push out means don't receive it now
            simulated_end_inventory = current_inventory_sim + simulated_supply - demand
            action = "Push Out"
            if suggested_po:
                used_po_docs.add(suggested_po)

        # ----------------------------------------------------

        results.append({
            'Snapshot Week': snapshot,
            'Start Inventory (Waterfall)': start_inventory_waterfall,
            'Start Inventory (Calc)': initial_inventory_calc if i == 0 else current_inventory_calc,
            'Demand (Waterfall)': demand,
            'Supply (Waterfall)': supply,
            'PO GR Quantity': po_received,
            'Purchasing Document(s)': ", ".join(map(str, po_docs_received)) if po_docs_received else None,
            'End Inventory (Waterfall)': end_inventory_waterfall,
            'End Inventory (Calc)': end_inventory_calc,
            'Flags': ", ".join(flags) if flags else "OK",
            'RCA Action': action,
            'Suggested PO for Action': suggested_po,
            'Adjusted GR WW': adjusted_gr_week,
            'End Inventory (Calc after Action)': simulated_end_inventory
        })

        # Advance inventory trackers
        current_inventory_calc = end_inventory_calc
        current_inventory_sim = simulated_end_inventory

    return pd.DataFrame(results)