import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import os
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def style_dataframe(filtered_df):
    # Identify week columns (WW12, WW13, etc.)
    ww_cols = [col for col in filtered_df.columns if col.startswith('WW')]

    # Define a row-wise styling function
    def highlight_row(row):
        lead_time = row['LeadTime(Week)']
        styles = []
        for col in row.index:
            if col in ww_cols:
                val = row[col]
                if pd.isna(val):
                    styles.append('')
                else:
                    try:
                        val = float(val)
                        if val < 0:
                            styles.append('background-color: #FF5252')  # Red
                        elif val < lead_time:
                            styles.append('background-color: #FFEB3B')  # Yellow
                        else:
                            styles.append('background-color: #4CAF50')  # Green
                    except:
                        styles.append('')
            else:
                styles.append('')
        return styles

    # Apply styling row by row
    styled_df = filtered_df.style.apply(highlight_row, axis=1)

    return styled_df

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

def extract_and_aggregate_weekly_data(folder_path, material_number, plant, site, start_week,cons_agg, num_weeks=12):
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

    result_df = adding_consumption_data_from_agg(result_df, cons_agg)

    # result_df = adding_consumption_data(result_df)

    return result_df, lead_value

def adding_consumption_data_from_agg(result_df, cons_agg):
    """
    Modifies result_df in-place by filling 'Consumption' rows using cons_agg.
    Each value is added on the diagonal (Snapshot == WW column), and other week columns are set to None.

    Args:
        result_df (pd.DataFrame): The target DataFrame with Snapshot and week columns.
        cons_agg (pd.DataFrame): DataFrame with 'WW' and 'Quantity' columns.
    
    Returns:
        pd.DataFrame: Updated DataFrame with Consumption values applied.
    """
    # Copy to avoid modifying in-place
    df = result_df.copy()

    # Get week columns
    week_cols = [col for col in df.columns if col.startswith("WW")]

    # Get example metadata row (first Supply or Demand row)
    meta_row = df[df['Measures'].isin(['Supply', 'Alternate Demand'])].iloc[0]

    # Store new rows
    new_rows = []

    for _, row in cons_agg.iterrows():
        week = row['WW']
        quantity = row['Quantity']

        if week not in week_cols:
            continue  # Skip invalid weeks

        # Create base row
        new_row = meta_row.copy()
        new_row['Measures'] = 'Consumption'
        new_row['Snapshot'] = week
        new_row['InventoryOn-Hand'] = None
        new_row['LeadTime(Week)'] = None

        # Set all week columns to None
        for col in week_cols:
            new_row[col] = None

        # Set quantity in diagonal cell
        new_row[week] = quantity

        new_rows.append(new_row)

    # Append all new consumption rows
    consumption_df = pd.DataFrame(new_rows)
    df = pd.concat([df, consumption_df], ignore_index=True)

    # Optional: sort to keep consistent order
    df.sort_values(by=['Snapshot', 'Measures'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def adding_consumption_data(df):
    # Step 1: Filter for Supply rows and sort by Snapshot
    supply_df = df[df['Measures'] == 'Supply'].copy()
    supply_df = supply_df.sort_values('Snapshot').reset_index(drop=True)

    # Step 2: Get list of week columns (e.g., WW12, WW13...)
    week_cols = [col for col in df.columns if col.startswith('WW')]

    # Step 3: Prepare output rows
    output_rows = []

    for i in range(len(supply_df) - 1):  # Skip last since no next week
        curr_row = supply_df.iloc[i]
        next_row = supply_df.iloc[i + 1]

        snapshot = curr_row['Snapshot']
        if snapshot not in week_cols:
            continue  # skip if the Snapshot name isn't a valid week column

        ioh_curr = curr_row['InventoryOn-Hand']
        supply_val = curr_row.get(snapshot, 0)
        ioh_next = next_row['InventoryOn-Hand']

        # Calculate consumption
        consumption = (ioh_curr + supply_val) - ioh_next

        # Create a new row with same meta data, but Measures = Consumption
        new_row = curr_row.copy()
        new_row['Measures'] = 'Consumption'
        new_row['InventoryOn-Hand'] = None

        # Populate week columns
        set_value = False
        for col in week_cols:
            if col == snapshot:
                new_row[col] = consumption
                set_value = True
            elif not set_value:
                new_row[col] = None
            else:
                new_row[col] = 0

        output_rows.append(new_row)

    # Step 4: Append to original df and sort
    consumption_df = pd.DataFrame(output_rows)
    combined_df = pd.concat([df, consumption_df], ignore_index=True)
    combined_df.sort_values(by=['Snapshot', 'Measures'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df



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

def apply_coloring_to_output(excel_buffer, lead_time, sheet_name):
    # Rewind buffer and load workbook
    excel_buffer.seek(0)
    wb = load_workbook(excel_buffer)
    ws = wb[sheet_name]

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

# Scenario 5: Identifying irregular consumption patterns
def analyze_week_to_week_demand_changes(result_df, abs_threshold=10, pct_threshold=0.3, lead_time = 6):
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

    # Find all WW columns
    ww_cols = [col for col in filtered_df.columns if col.startswith("WW")]

    output_all = []

    lead_time = lead_time.iloc[0] #lead time taking in as a series

    for i in range(1, len(ww_cols)):  # start from 2nd WW column
        current_col = ww_cols[i]

        # Determine the window size
        window_size = i + 1

        # Determine row indices
        if i < (lead_time+1):
            window_data = filtered_df[current_col].iloc[:window_size]
            snapshot_ids = filtered_df.index[:window_size]
        else:
            window_data = filtered_df[current_col].iloc[-(lead_time+1):]
            snapshot_ids = filtered_df.index[-(lead_time+1):]

        # Build temporary DataFrame
        output_df = pd.DataFrame({
            "Snapshot": snapshot_ids,
            "Demand w/o Buffer": window_data.values,
            "Week": current_col
        })

        # Sort and calculate WoW change
        output_df = output_df.sort_values(by="Snapshot").reset_index(drop=True)
        output_df["WoW Change"] = output_df["Demand w/o Buffer"].diff()
        output_df["WoW % Change"] = output_df["Demand w/o Buffer"].pct_change()

        # Flag spikes and drops
        output_df["Spike"] = output_df["WoW Change"] > abs_threshold
        output_df["Drop"] = output_df["WoW Change"] < -abs_threshold
        output_df["Sudden % Spike"] = output_df["WoW % Change"] > pct_threshold
        output_df["Sudden % Drop"] = output_df["WoW % Change"] < -pct_threshold

        # Store for analysis
        output_all.append(output_df)

    # Combine all weeks’ flagged data
    final_output_df = pd.concat(output_all, ignore_index=True)
    final_output_df = final_output_df.drop(columns=['Snapshot'])
    cols = final_output_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Week')))
    final_output_df = final_output_df[cols]

    return final_output_df


def scenario_6(waterfall_df, po_df):
    # Filter relevant rows
    supply_rows = waterfall_df[waterfall_df['Measures'] == 'Supply']
    demand_rows = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer']
    consumption_rows = waterfall_df[waterfall_df['Measures'] == 'Consumption']

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

        # Get consumption value
        consumption_val = consumption_rows[consumption_rows['Snapshot'] == snapshot][week_col]
        consumption = int(consumption_val.values[0]) if not consumption_val.empty else 0

        # Start inventory from Waterfall column
        inv_val = supply_rows[supply_rows['Snapshot'] == snapshot]['InventoryOn-Hand']
        start_inventory_waterfall = int(inv_val.values[0]) if not inv_val.empty else 0

        # GR quantity from PO data
        po_received = 0
        if not po_df.empty:
            po_received = po_df[po_df['GR WW'] == week_num]['GR Quantity'].sum()

        # Calculate end inventories INCLUDING PO received
        end_inventory_calc = current_inventory_calc + po_received - demand
        end_inventory_waterfall = start_inventory_waterfall + supply - demand

        # Check for irregular consumption patterns
        irregular_pattern = None
        if consumption < 0:
            if results:
                previous_week = results[-1]
                prev_end_inv = previous_week['End Inventory (Waterfall)']
                prev_demand = previous_week['Demand (Waterfall)']
                prev_consumption = previous_week['Consumption (Waterfall)']
                prev_supply = previous_week['Supply (Waterfall)']
                prev_snapshot = previous_week['Snapshot Week']

                irregular_pattern = (
                    f"In week {snapshot}, the reported consumption was negative, which suggests a possible inventory correction or data anomaly. "
                    f"Looking back, in week {prev_snapshot}, the starting inventory was {previous_week['Start Inventory (Waterfall)']}, "
                    f"supply was {prev_supply}, and demand was {prev_demand}, resulting in an expected end inventory of {prev_end_inv}. "
                    f"However, the current week's starting inventory ({start_inventory_waterfall}) is higher than the expected ending inventory from last week, "
                    f"which implies inventory was added back — potentially due to returns, cancellations, or adjustments outside the normal flow."
                )
            else:
                irregular_pattern = (
                    f"In week {snapshot}, negative consumption was reported. Since this is the first week in the timeline, "
                    f"it may indicate an opening adjustment or return to inventory that wasn't accounted for in demand."
                )
        elif consumption > demand:
            irregular_pattern = "More consumption than demand"
        elif consumption == 0 and demand != 0:
            irregular_pattern = "Consumption is zero but demand is not"
        elif consumption != 0 and demand == 0:
            irregular_pattern = "Demand is zero but consumption is not"
        else:
            pass

        # Record results
        results.append({
            'Snapshot Week': snapshot,
            'Start Inventory (Waterfall)': start_inventory_waterfall,
            'Start Inventory (Calc)': current_inventory_calc,
            'Demand (Waterfall)': demand,
            'Supply (Waterfall)': supply,
            'Consumption (Waterfall)': consumption,
            'PO GR Quantity': po_received,
            'End Inventory (Waterfall)': end_inventory_waterfall,
            'End Inventory (Calc)': end_inventory_calc,
            'Irregular Pattern': irregular_pattern
        })

        # Update for next loop
        current_inventory_calc = end_inventory_calc

    return pd.DataFrame(results)

def scenario_1(df, po_df):
    # Filter for 'Weeks of Stock' rows
    weeks_df = df[df['Measures'] == 'Weeks of Stock'].copy()
    weeks_df = weeks_df.reset_index(drop=True)

    # Get all WW columns and ensure they're ordered
    week_cols = sorted([col for col in df.columns if col.startswith('WW')])

    # Function to get week range per row
    def get_leadtime_cols(snapshot, leadtime, all_weeks):
        if snapshot not in all_weeks:
            return []
        start_idx = all_weeks.index(snapshot)
        end_idx = start_idx + leadtime  # snapshot + (leadtime - 1)
        return all_weeks[start_idx:end_idx]

    # Create filtered output
    filtered_rows = []
    for _, row in weeks_df.iterrows():
        leadtime = int(row['LeadTime(Week)'])
        snapshot = row['Snapshot']
        cols_to_keep = get_leadtime_cols(snapshot, leadtime, week_cols)

        base_info = row.drop(week_cols + ['InventoryOn-Hand'], errors='ignore')  # Keep non-WW fields
        week_data = row[cols_to_keep]   # Select WW columns in range
        combined = pd.concat([base_info, week_data])
        filtered_rows.append(combined)

    # Final filtered DataFrame
    filtered_df = pd.DataFrame(filtered_rows)

    # Ensure Snapshot and GR WW are numeric and comparable
    po_df['GR WW'] = po_df['GR WW'].astype(int)
    filtered_df['Snapshot'] = filtered_df['Snapshot'].str.extract(r'(\d+)').astype(int)

    # Function to filter POs based on lead time
    def filter_pos_by_leadtime(row, leadtime, po_df):
        snapshot = row['Snapshot']
        start_week = snapshot
        end_week = snapshot + leadtime - 1
        gr_weeks = list(range(start_week, end_week + 1))
        filtered_po_df = po_df[po_df['GR WW'].isin(gr_weeks)]
        incoming_po = ', '.join(filtered_po_df['Purchasing Document'].astype(str).unique())
        return incoming_po

    # Add Incoming PO column
    filtered_df['Incoming PO'] = filtered_df.apply(
        lambda row: filter_pos_by_leadtime(row, row['LeadTime(Week)'], po_df), axis=1
    )

    def flag_row_with_reason(row):
        leadtime = row['LeadTime(Week)']
        has_incoming_po = row['Incoming PO'] != ''
        week_cols_in_row = [col for col in week_cols if col in row.index]
        values_series = row[week_cols_in_row]
        numeric_values = values_series[values_series.notna()].astype(float)

        if numeric_values.empty:
            return 'Unknown Case - All Weeks Null', 'All Weeks of Stock values are missing or null.'

        below_lt = numeric_values < leadtime
        negative = numeric_values < 0
        adequate = (numeric_values >= leadtime) & (numeric_values >= 0)

        if below_lt.all() or negative.all():
            if has_incoming_po:
                return 'Adequate', 'Stock is low but incoming PO exists within lead time.'
            else:
                return 'Inadequate', 'All WoS values are below lead time and no PO is expected.'
        elif adequate.all():
            return 'Not Applicable', 'All WoS values meet or exceed lead time—no issue.'
        elif below_lt.any() or negative.any():
            if has_incoming_po:
                return 'Partially Adequate', 'Some WoS values are low, but there is an incoming PO.'
            else:
                return 'Partially Inadequate', 'Some WoS values are low and there is no incoming PO.'
        else:
            return 'Unknown Case - Numeric Checks Failed', 'Unexpected data condition—please review values.'

    # Apply and separate into two columns
    filtered_df[['Flag', 'Reason']] = filtered_df.apply(
        flag_row_with_reason, axis=1, result_type='expand'
    )

    return filtered_df


def old_scenario_2(waterfall_df, po_df):
    supply_rows = waterfall_df[waterfall_df['Measures'] == 'Supply']
    demand_rows = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer']

    initial_snapshot = supply_rows['Snapshot'].iloc[0]
    initial_inventory_calc = int(supply_rows[supply_rows['Snapshot'] == initial_snapshot]['InventoryOn-Hand'].values[0])

    snapshots = waterfall_df['Snapshot'].unique()
    results = []

    current_inventory_calc = initial_inventory_calc
    current_inventory_sim = initial_inventory_calc
    simulated_gr_schedule = {}  # key: week, value: list of GR quantities arriving
    used_po_docs = set()

    for i, snapshot in enumerate(snapshots):
        week_col = snapshot
        week_num = int(snapshot.replace("WW", ""))

        # Demand & Supply
        demand_val = demand_rows[demand_rows['Snapshot'] == snapshot][week_col]
        supply_val = supply_rows[supply_rows['Snapshot'] == snapshot][week_col]
        demand = int(demand_val.values[0]) if not demand_val.empty else 0
        supply = int(supply_val.values[0]) if not supply_val.empty else 0

        # Waterfall Inventory
        inv_val = supply_rows[supply_rows['Snapshot'] == snapshot]['InventoryOn-Hand']
        start_inventory_waterfall = int(inv_val.values[0]) if not inv_val.empty else 0

        # Actual PO receipts
        po_received_data = po_df[(po_df['GR WW'] == week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))]
        po_received = po_received_data['GR Quantity'].sum()
        po_docs_received = list(po_received_data['Purchasing Document'])
        print(po_docs_received)

        # Inventory (Actual)
        end_inventory_calc = current_inventory_calc + supply + po_received - demand
        end_inventory_waterfall = start_inventory_waterfall + supply + po_received - demand

        # --- Simulation Setup ---
        simulated_po_received = sum(simulated_gr_schedule.get(week_num, []))
        simulated_supply = supply
        start_inventory_sim = current_inventory_sim + simulated_po_received
        simulated_end_inventory = start_inventory_sim + simulated_supply + po_received - demand

        # --- Action Planning ---
        flags = []
        action = "No Action"
        suggested_pos = []
        adjusted_gr_weeks = []
        po_qty_ordered_total = 0
        po_gr_after_action = 0

        # Pull-in logic for negative inventory
        if simulated_end_inventory < 0:
            flags.append("Inventory went negative — stockout")
            shortfall = abs(simulated_end_inventory)

            future_po_candidates = po_df[
                (po_df['GR WW'] > week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))
            ].sort_values(by=['GR WW', 'Order WW'])

            for _, po in future_po_candidates.iterrows():
                lead_time = po['GR WW'] - po['Order WW']
                adjusted_gr_week = week_num + lead_time

                # Skip if already too late even after pull-in
                if adjusted_gr_week > week_num:
                    continue

                po_qty = po['GR Quantity']
                doc_num = po['Purchasing Document']

                simulated_gr_schedule.setdefault(adjusted_gr_week, []).append(po_qty)
                used_po_docs.add(doc_num)

                suggested_pos.append(doc_num)
                adjusted_gr_weeks.append(f"WW{adjusted_gr_week}")
                po_qty_ordered_total += po_qty
                po_gr_after_action += po_qty

                shortfall -= po_qty
                if shortfall <= 0:
                    break

            if suggested_pos:
                action = "Pull In"
                simulated_end_inventory = start_inventory_sim + simulated_supply + po_received + po_gr_after_action - demand

        # Push-out logic for excess inventory
        elif simulated_end_inventory > current_inventory_sim + supply - demand and po_received > 0:
            flags.append("Inventory built up unnecessarily — potential overstock")
            suggested_po = po_docs_received[0] if po_docs_received else None
            if suggested_po:
                po_qty_ordered_total = po_received_data[po_received_data['Purchasing Document'] == suggested_po]['GR Quantity'].values[0]
                suggested_pos.append(suggested_po)
                adjusted_gr_weeks.append(f"WW{week_num + 1}")
                po_gr_after_action = 0
                simulated_end_inventory = current_inventory_sim + supply - demand
                used_po_docs.add(suggested_po)
                action = "Push Out"

        results.append({
            'Snapshot Week': snapshot,
            'Start Inventory (Waterfall)': start_inventory_waterfall,
            'Start Inventory (Calc)': initial_inventory_calc if i == 0 else current_inventory_calc,
            'Start Inventory (Calc after Action)': start_inventory_sim,
            'Demand (Waterfall)': demand,
            'Supply (Waterfall)': supply,
            'PO GR Quantity': po_received,
            'PO GR Quantity (After Action)': po_gr_after_action,
            'End Inventory (Waterfall)': end_inventory_waterfall,
            'End Inventory (Calc)': end_inventory_calc,
            'End Inventory (Calc after Action)': simulated_end_inventory,
            'Purchasing Document(s)': ", ".join(map(str, po_docs_received)) if po_docs_received else None,
            'Suggested PO for Action': ", ".join(map(str, suggested_pos)) if suggested_pos else None,
            'PO Quantity Ordered': po_qty_ordered_total,
            'Adjusted GR WW': ", ".join(adjusted_gr_weeks) if adjusted_gr_weeks else None,
            'Flags': ", ".join(flags) if flags else "OK",
            'RCA Action': action
        })

        current_inventory_calc = end_inventory_calc
        current_inventory_sim = simulated_end_inventory

    return pd.DataFrame(results)

def scenario_2_old_v2(waterfall_df, po_df, lead_time=6):
    supply_rows = waterfall_df[waterfall_df['Measures'] == 'Supply']
    demand_rows = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer']

    initial_snapshot = supply_rows['Snapshot'].iloc[0]
    initial_inventory_calc = int(supply_rows[supply_rows['Snapshot'] == initial_snapshot]['InventoryOn-Hand'].values[0])

    snapshots = waterfall_df['Snapshot'].unique()
    snapshot_to_week = {s: int(s.replace("WW", "")) for s in snapshots}
    week_to_snapshot = {v: k for k, v in snapshot_to_week.items()}
    results = []

    current_inventory_calc = initial_inventory_calc
    current_inventory_sim = initial_inventory_calc
    simulated_gr_schedule = {}  # key: week, value: list of GR quantities arriving
    used_po_docs = set()

    for i, snapshot in enumerate(snapshots):
        week_col = snapshot
        week_num = snapshot_to_week[snapshot]

        # Demand & Supply values
        demand_val = demand_rows[demand_rows['Snapshot'] == snapshot][week_col]
        supply_val = supply_rows[supply_rows['Snapshot'] == snapshot][week_col]
        demand = int(demand_val.values[0]) if not demand_val.empty else 0
        supply = int(supply_val.values[0]) if not supply_val.empty else 0

        # Waterfall Inventory
        inv_val = supply_rows[supply_rows['Snapshot'] == snapshot]['InventoryOn-Hand']
        start_inventory_waterfall = int(inv_val.values[0]) if not inv_val.empty else 0

        # Actual PO receipts
        po_received_data = po_df[(po_df['GR WW'] == week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))]
        po_received = po_received_data['GR Quantity'].sum()
        po_docs_received = list(po_received_data['Purchasing Document'])

        # Inventory (Actual)
        end_inventory_calc = current_inventory_calc + supply + po_received - demand
        end_inventory_waterfall = start_inventory_waterfall + supply + po_received - demand

        # Simulation setup
        simulated_po_received = sum(simulated_gr_schedule.get(week_num, []))
        start_inventory_sim = current_inventory_sim + simulated_po_received
        simulated_end_inventory = start_inventory_sim + supply + po_received - demand

        # --- LT-based Demand/Supply Check ---
        lt_weeks = [week_num + offset for offset in range(lead_time)]
        lt_snapshots = [week_to_snapshot[w] for w in lt_weeks if w in week_to_snapshot]

        total_lt_demand = demand_rows[demand_rows['Snapshot'].isin(lt_snapshots)][lt_snapshots].sum(axis=1).sum()
        total_lt_supply = (
            supply_rows[supply_rows['Snapshot'].isin(lt_snapshots)][lt_snapshots].sum(axis=1).sum() +
            po_df[(po_df['GR WW'].isin(lt_weeks)) & (~po_df['Purchasing Document'].isin(used_po_docs))]['GR Quantity'].sum()
        )

        flags = []
        action = "No Action"
        suggested_pos = []
        adjusted_gr_weeks = []
        po_qty_ordered_total = 0
        po_gr_after_action = 0

        # Pull-in logic
        if total_lt_demand > total_lt_supply:
            flags.append("Insufficient supply within lead time")
            shortfall = total_lt_demand - total_lt_supply

            future_po_candidates = po_df[
                (po_df['GR WW'] > week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))
            ].sort_values(by=['GR WW', 'Order WW'])

            for _, po in future_po_candidates.iterrows():
                po_qty = po['GR Quantity']
                doc_num = po['Purchasing Document']
                lead_time_of_po = po['GR WW'] - po['Order WW']
                adjusted_gr_week = week_num + lead_time_of_po

                if adjusted_gr_week > week_num + lead_time - 1:
                    continue  # Can't make it in time

                simulated_gr_schedule.setdefault(adjusted_gr_week, []).append(po_qty)
                used_po_docs.add(doc_num)

                suggested_pos.append(doc_num)
                adjusted_gr_weeks.append(f"WW{adjusted_gr_week}")
                po_qty_ordered_total += po_qty
                po_gr_after_action += po_qty

                shortfall -= po_qty
                if shortfall <= 0:
                    break

            if suggested_pos:
                action = "Pull In"
                simulated_end_inventory = start_inventory_sim + supply + po_received + po_gr_after_action - demand

        # Push-out logic
        elif total_lt_supply > total_lt_demand and po_received > 0:
            flags.append("Excess supply within lead time")
            suggested_po = po_docs_received[0] if po_docs_received else None
            if suggested_po:
                po_qty_ordered_total = po_received_data[po_received_data['Purchasing Document'] == suggested_po]['GR Quantity'].values[0]
                suggested_pos.append(suggested_po)
                adjusted_gr_weeks.append(f"WW{week_num + 1}")
                po_gr_after_action = 0
                simulated_end_inventory = current_inventory_sim + supply - demand
                used_po_docs.add(suggested_po)
                action = "Push Out"

        results.append({
            'Snapshot Week': snapshot,
            'Start Inventory (Waterfall)': start_inventory_waterfall,
            'Start Inventory (Calc)': initial_inventory_calc if i == 0 else current_inventory_calc,
            'Start Inventory (Calc after Action)': start_inventory_sim,
            'Demand (Waterfall)': demand,
            'Supply (Waterfall)': supply,
            'PO GR Quantity': po_received,
            'PO GR Quantity (After Action)': po_gr_after_action,
            'End Inventory (Waterfall)': end_inventory_waterfall,
            'End Inventory (Calc)': end_inventory_calc,
            'End Inventory (Calc after Action)': simulated_end_inventory,
            'Purchasing Document(s)': ", ".join(map(str, po_docs_received)) if po_docs_received else None,
            'Suggested PO for Action': ", ".join(map(str, suggested_pos)) if suggested_pos else None,
            'PO Quantity Ordered': po_qty_ordered_total,
            'Adjusted GR WW': ", ".join(adjusted_gr_weeks) if adjusted_gr_weeks else None,
            'Flags': ", ".join(flags) if flags else "OK",
            'RCA Action': action
        })

        current_inventory_calc = end_inventory_calc
        current_inventory_sim = simulated_end_inventory

    return pd.DataFrame(results)

def scenario_2(waterfall_df, po_df, start_week):
    """
    Simulates inventory levels, plans actions (pull-in/push-out), and provides
    supply-demand adequacy recommendations based on lead time.

    Args:
        waterfall_df (pd.DataFrame): DataFrame containing supply, demand,
                                     inventory, and lead time data across snapshots.
                                     Expected columns: 'Measures', 'Snapshot',
                                     'InventoryOn-Hand', 'LeadTime(Week)',
                                     and 'WWXX' columns for supply/demand values.
        po_df (pd.DataFrame): DataFrame containing Purchase Order (PO) data.
                              Expected columns: 'Purchasing Document', 'Order WW',
                              'GR WW', 'GR Quantity'.
        start_week (int): The starting week number (e.g., 2 for WW02) from
                          which the supply-demand adequacy recommendations should be generated.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - df_inventory_actions (pd.DataFrame): Results of inventory calculation
                                                     and action planning.
               - df_recommendations (pd.DataFrame): Supply-demand adequacy recommendations
                                                    based on lead time.
    """
    # --- Diagnostic: Check 'Snapshot' column type before processing ---
    print("--- Debugging 'Snapshot' Column ---")
    print(f"Original 'Snapshot' column dtype: {waterfall_df['Snapshot'].dtype}")
    print(f"Original 'Snapshot' column head:\n{waterfall_df['Snapshot'].head()}")
    print("-----------------------------------")
    # -------------------------------------------------------------------

    # Create copies and ensure 'Snapshot' is string type before adding 'WeekNum'
    supply_rows_all = waterfall_df[waterfall_df['Measures'] == 'Supply'].copy()
    supply_rows_all['Snapshot'] = supply_rows_all['Snapshot'].astype(str) # Convert Snapshot column in this copy
    supply_rows_all['WeekNum'] = supply_rows_all['Snapshot'].str.replace('WW', '').astype(int)

    demand_rows_all = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer'].copy()
    demand_rows_all['Snapshot'] = demand_rows_all['Snapshot'].astype(str) # Convert Snapshot column in this copy
    demand_rows_all['WeekNum'] = demand_rows_all['Snapshot'].str.replace('WW', '').astype(int)

    waterfall_df_with_weeknum = waterfall_df.copy()
    waterfall_df_with_weeknum['Snapshot'] = waterfall_df_with_weeknum['Snapshot'].astype(str) # Convert Snapshot column in this copy
    waterfall_df_with_weeknum['WeekNum'] = waterfall_df_with_weeknum['Snapshot'].str.replace('WW', '').astype(int)

    # Get unique snapshots, ensuring they are strings
    snapshots = waterfall_df['Snapshot'].astype(str).unique()
    
    # Initial inventory from the very first snapshot's supply row
    initial_snapshot = supply_rows_all['Snapshot'].iloc[0]
    initial_inventory_calc = int(supply_rows_all[supply_rows_all['Snapshot'] == initial_snapshot]['InventoryOn-Hand'].values[0])

    results_inventory_actions = []  # For inventory calculation and action planning
    results_recommendations = []    # For new supply-demand adequacy recommendations

    current_inventory_calc = initial_inventory_calc
    current_inventory_sim = initial_inventory_calc  # Tracks the simulated end inventory from the previous week

    simulated_gr_schedule = {}  # key: week_num, value: list of GR quantities arriving due to past actions (e.g., push-outs)
    used_po_docs = set()        # Tracks Purchasing Documents that have been acted upon (pulled in or pushed out)

    for i, snapshot in enumerate(snapshots):
        # Ensure snapshot is treated as string before operations
        week_col = str(snapshot)  # e.g., 'WW01'
        week_num = int(week_col.replace("WW", ""))  # e.g., 1

        # --- Inventory Calculation and Action Planning (Existing Logic) ---
        # Get current week's demand and supply from waterfall_df
        demand_val = demand_rows_all[demand_rows_all['Snapshot'] == week_col][week_col]
        supply_val = supply_rows_all[supply_rows_all['Snapshot'] == week_col][week_col]
        demand = int(demand_val.values[0]) if not demand_val.empty else 0
        supply = int(supply_val.values[0]) if not supply_val.empty else 0

        # Get current week's InventoryOn-Hand from waterfall_df (for Waterfall column in output)
        inv_val = supply_rows_all[supply_rows_all['Snapshot'] == week_col]['InventoryOn-Hand']
        start_inventory_waterfall = int(inv_val.values[0]) if not inv_val.empty else 0

        # Actual PO receipts for the current week
        po_received_data = po_df[(po_df['GR WW'] == week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))]
        po_received_current_week = po_received_data['GR Quantity'].sum()
        po_docs_received_current_week = list(po_received_data['Purchasing Document'])
        
        # Calculate end inventory (Calc - direct from waterfall values + actual POs)
        end_inventory_calc = current_inventory_calc + supply + po_received_current_week - demand
        end_inventory_waterfall = start_inventory_waterfall + supply + po_received_current_week - demand

        # --- Simulation Setup for Current Week ---
        simulated_pos_arriving_this_week = sum(simulated_gr_schedule.get(week_num, []))
        start_inventory_sim_before_action = current_inventory_sim + simulated_pos_arriving_this_week
        simulated_end_inventory = start_inventory_sim_before_action + supply + po_received_current_week - demand

        # --- Action Planning for Current Week ---
        flags = []
        action = "No Action"
        suggested_pos = []
        adjusted_gr_weeks = []
        po_qty_ordered_total = 0
        po_gr_after_action = po_received_current_week

        # Pull-in logic
        if simulated_end_inventory < 0:
            flags.append("Inventory went negative — stockout")
            shortfall = abs(simulated_end_inventory)
            future_po_candidates = po_df[
                (po_df['GR WW'] > week_num) & (~po_df['Purchasing Document'].isin(used_po_docs))
            ].sort_values(by=['GR WW', 'Order WW'])

            for _, po in future_po_candidates.iterrows():
                po_qty = po['GR Quantity']
                doc_num = po['Purchasing Document']
                simulated_gr_schedule.setdefault(week_num, []).append(po_qty)
                used_po_docs.add(doc_num)
                suggested_pos.append(doc_num)
                adjusted_gr_weeks.append(f"WW{week_num:02d}")
                po_qty_ordered_total += po_qty
                shortfall -= po_qty
                if shortfall <= 0: break

            if suggested_pos:
                action = "Pull In"
                # Recalculate simulated_end_inventory after pull-in
                simulated_end_inventory = current_inventory_sim + sum(simulated_gr_schedule.get(week_num, [])) + supply + po_received_current_week - demand
                po_gr_after_action = po_received_current_week + po_qty_ordered_total

        # --- Push-out Logic: Check for potential overstock based on demand trajectory ---
        # Get current week's lead time for overstock check
        current_lead_time_row = supply_rows_all[supply_rows_all['Snapshot'] == week_col]
        lead_time_val = int(current_lead_time_row['LeadTime(Week)'].iloc[0]) if not current_lead_time_row.empty and 'LeadTime(Week)' in current_lead_time_row.columns else 0

        # Define a buffer period for overstock calculation (e.g., another lead time or a fixed number of weeks)
        buffer_weeks_for_overstock = lead_time_val # User wants "another 6 weeks", implying same as lead time

        # Calculate total demand and original waterfall supply in the extended horizon
        # The extended horizon spans from `week_num` to `week_num + lead_time_val + buffer_weeks_for_overstock`
        extended_horizon_end_week = week_num + lead_time_val + buffer_weeks_for_overstock
        
        total_demand_in_extended_horizon = 0
        total_waterfall_supply_in_extended_horizon = 0 # This represents original waterfall supply

        # Get the row for the current snapshot from waterfall_df (which contains all WW columns)
        # Use waterfall_df_with_weeknum to access the 'WeekNum' column
        current_snapshot_supply_row = waterfall_df_with_weeknum[
            (waterfall_df_with_weeknum['Measures'] == 'Supply') &
            (waterfall_df_with_weeknum['WeekNum'] == week_num)
        ]
        current_snapshot_demand_row = waterfall_df_with_weeknum[
            (waterfall_df_with_weeknum['Measures'] == 'Demand w/o Buffer') &
            (waterfall_df_with_weeknum['WeekNum'] == week_num)
        ]

        for horizon_week_num in range(week_num, extended_horizon_end_week + 1):
            col_name = f"WW{horizon_week_num:02d}"
            
            if not current_snapshot_demand_row.empty and col_name in current_snapshot_demand_row.columns:
                demand_val_in_horizon = current_snapshot_demand_row[col_name].iloc[0]
                total_demand_in_extended_horizon += int(demand_val_in_horizon)
            
            if not current_snapshot_supply_row.empty and col_name in current_snapshot_supply_row.columns:
                supply_val_in_horizon = current_snapshot_supply_row[col_name].iloc[0]
                total_waterfall_supply_in_extended_horizon += int(supply_val_in_horizon)
        
        # Calculate average weekly demand over the extended horizon (to avoid division by zero later)
        num_weeks_in_extended_horizon = extended_horizon_end_week - week_num + 1
        average_weekly_demand_extended = 0
        if num_weeks_in_extended_horizon > 0:
            average_weekly_demand_extended = total_demand_in_extended_horizon / num_weeks_in_extended_horizon

        # Calculate the total available resource for covering the extended horizon demand
        # This is the `simulated_end_inventory` (at end of current week)
        # PLUS future waterfall supply (from next week onwards in the extended horizon)
        
        # The `simulated_end_inventory` already includes the current week's `supply` from waterfall.
        # So, `total_waterfall_supply_in_extended_horizon` needs to be adjusted to exclude current week's `supply`.
        total_future_waterfall_supply_from_next_week = total_waterfall_supply_in_extended_horizon - supply

        total_resource_for_extended_horizon = simulated_end_inventory + total_future_waterfall_supply_from_next_week

        # Define a safety buffer for overstock (e.g., if supply can last X weeks beyond required horizon)
        safety_weeks_buffer_for_overstock = 1 # e.g., 1 week buffer beyond the extended horizon

        # Overstock condition: If current inventory + future supply can cover more than the
        # extended horizon's demand PLUS a safety buffer, AND there are POs this week to push out.
        if average_weekly_demand_extended > 0:
            # Calculate how many weeks of demand the total resource can cover
            weeks_of_coverage_by_resource = total_resource_for_extended_horizon / average_weekly_demand_extended
            
            # Calculate the required weeks of coverage (extended horizon duration + safety buffer)
            required_weeks_coverage = num_weeks_in_extended_horizon + safety_weeks_buffer_for_overstock

            if weeks_of_coverage_by_resource > required_weeks_coverage and po_received_current_week > 0:
                flags.append(f"Inventory built up unnecessarily — potential overstock (Supply covers {weeks_of_coverage_by_resource:.1f} weeks; required {required_weeks_coverage} weeks)")
                
                # Push-out logic:
                if po_docs_received_current_week:
                    # Select a PO to push out. Let's pick the one that was originally scheduled latest or ordered earliest
                    # We sort by GR WW descending, then Order WW descending, to prioritize pushing out POs that arrived later.
                    po_to_push = po_received_data.sort_values(by=['GR WW', 'Order WW'], ascending=[False, False]).iloc[0]
                    suggested_po_to_push = po_to_push['Purchasing Document']
                    po_qty_to_push_out = po_to_push['GR Quantity']
                    
                    suggested_pos.append(suggested_po_to_push)
                    # Push out to the next week
                    adjusted_gr_weeks.append(f"WW{week_num + 1:02d}")
                    po_qty_ordered_total = po_qty_to_push_out
                    
                    # Adjust current week's received POs for simulation
                    po_gr_after_action = po_received_current_week - po_qty_to_push_out
                    
                    # Add the pushed-out quantity to the simulated GR schedule for the next week
                    simulated_gr_schedule.setdefault(week_num + 1, []).append(po_qty_to_push_out)
                    used_po_docs.add(suggested_po_to_push)
                    
                    # Recalculate simulated_end_inventory after the push-out action
                    simulated_end_inventory = current_inventory_sim + simulated_pos_arriving_this_week + po_gr_after_action + supply - demand
                    action = "Push Out"
        # --- End of Push-out Logic ---

        start_inventory_sim_final = current_inventory_sim + sum(simulated_gr_schedule.get(week_num, []))

        results_inventory_actions.append({
            'Snapshot Week': week_col, # Use week_col (string format) for consistency in output
            'Start Inventory (Waterfall)': start_inventory_waterfall,
            'Start Inventory (Calc)': initial_inventory_calc if i == 0 else current_inventory_calc,
            'Start Inventory (Calc after Action)': start_inventory_sim_final,
            'Demand (Waterfall)': demand,
            'Supply (Waterfall)': supply,
            'PO GR Quantity': po_received_current_week,
            'PO GR Quantity (After Action)': po_gr_after_action,
            'End Inventory (Waterfall)': end_inventory_waterfall,
            'End Inventory (Calc)': end_inventory_calc,
            'End Inventory (Calc after Action)': simulated_end_inventory,
            'Purchasing Document(s)': ", ".join(map(str, po_docs_received_current_week)) if po_docs_received_current_week else None,
            'Suggested PO for Action': ", ".join(map(str, suggested_pos)) if suggested_pos else None,
            'PO Quantity Ordered': po_qty_ordered_total,
            'Adjusted GR WW': ", ".join(adjusted_gr_weeks) if adjusted_gr_weeks else None,
            'Flags': ", ".join(flags) if flags else "OK",
            'RCA Action': action
        })

        current_inventory_calc = end_inventory_calc
        current_inventory_sim = simulated_end_inventory

    # --- Supply-Demand Adequacy Recommendation Logic (Remains largely unchanged) ---
    for snapshot in snapshots:
        current_week_num = int(str(snapshot).replace("WW", "")) # Ensure snapshot is string

        if current_week_num < start_week:
            continue

        current_lead_time_row = supply_rows_all[supply_rows_all['Snapshot'] == str(snapshot)] # Filter using string snapshot
        lead_time_val = int(current_lead_time_row['LeadTime(Week)'].iloc[0]) if not current_lead_time_row.empty and 'LeadTime(Week)' in current_lead_time_row.columns else 0

        look_ahead_end_week_num = current_week_num + lead_time_val

        total_supply_within_lt = 0
        total_demand_within_lt = 0

        supply_row_for_snapshot = waterfall_df[
            (waterfall_df['Measures'] == 'Supply') &
            (waterfall_df['Snapshot'].astype(str) == str(snapshot)) # Convert to string for comparison
        ]
        demand_row_for_snapshot = waterfall_df[
            (waterfall_df['Measures'] == 'Demand w/o Buffer') &
            (waterfall_df['Snapshot'].astype(str) == str(snapshot)) # Convert to string for comparison
        ]
        
        for week_in_horizon_num in range(current_week_num, look_ahead_end_week_num + 1):
            col_name_for_horizon_week = f"WW{week_in_horizon_num:02d}"
            
            if not supply_row_for_snapshot.empty and col_name_for_horizon_week in supply_row_for_snapshot.columns:
                supply_val_in_horizon = supply_row_for_snapshot[col_name_for_horizon_week].iloc[0]
                total_supply_within_lt += int(supply_val_in_horizon)
            
            if not demand_row_for_snapshot.empty and col_name_for_horizon_week in demand_row_for_snapshot.columns:
                demand_val_in_horizon = demand_row_for_snapshot[col_name_for_horizon_week].iloc[0]
                total_demand_within_lt += int(demand_val_in_horizon)

        adequate_flag = "Shortfall"
        if total_supply_within_lt >= total_demand_within_lt:
            adequate_flag = "Adequate"
        
        results_recommendations.append({
            'Snapshot Week': str(snapshot), # Use string format for consistency
            'Lead Time (Weeks)': lead_time_val,
            'Look Ahead End Week': f"WW{look_ahead_end_week_num:02d}",
            'Total Supply within LT': total_supply_within_lt,
            'Total Demand within LT': total_demand_within_lt,
            'Adequacy Flag': adequate_flag
        })

    df_inventory_actions = pd.DataFrame(results_inventory_actions)
    df_recommendations = pd.DataFrame(results_recommendations)

    return df_inventory_actions, df_recommendations



def scenario_3(waterfall_df, po_df, scenario_1_results_df):
    """
    Analyzes future supply vs. demand based on confirmed POs and waterfall,
    using the final calculated inventory from check_demand1 results as the starting point,
    and suggests potential PO adjustments, including relevant PO numbers.

    Args:
        waterfall_df (pd.DataFrame): DataFrame containing waterfall data
                                   (Measures: 'Supply', 'Demand w/o Buffer').
                                   Expected columns: 'Measures', 'Snapshot', 'WWxx' columns.
        po_df (pd.DataFrame): DataFrame containing PO data with future delivery dates.
                            Expected columns: 'Delivery WW', 'PO Quantity', 'Purchasing Document'.
        scenario_1_results_df (pd.DataFrame): The output DataFrame from the scenario_1 function.
                                            Expected column: 'End Inventory (Calc)'.

    Returns:
        pd.DataFrame: Analysis results including projected inventory and potential flags/suggestions
                      with PO numbers.
    """
    # --- Extract Starting Inventory from Scenario 1 Results ---
    starting_inventory = 0 # Default
    if scenario_1_results_df is None or scenario_1_results_df.empty or 'End Inventory (Calc)' not in scenario_1_results_df.columns:
        print("Warning: Could not determine starting inventory from scenario_1_results_df. Starting with 0.")
    else:
        last_inv_calc = scenario_1_results_df['End Inventory (Calc)'].iloc[-1]
        if pd.isna(last_inv_calc):
             print("Warning: Last 'End Inventory (Calc)' from scenario_1_results_df is NaN. Starting with 0.")
             starting_inventory = 0
        else:
             starting_inventory = int(last_inv_calc)

    print(f"Starting Scenario 2 with inventory: {starting_inventory}")
    # --- End Extraction ---


    # Filter relevant rows from waterfall for the latest plan
    latest_demand_plan_row = waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer'].iloc[-1] if not waterfall_df[waterfall_df['Measures'] == 'Demand w/o Buffer'].empty else None
    latest_supply_plan_row = waterfall_df[waterfall_df['Measures'] == 'Supply'].iloc[-1] if not waterfall_df[waterfall_df['Measures'] == 'Supply'].empty else None

    # Identify future snapshot columns (WWxx) and sort them chronologically
    snapshot_cols = [col for col in waterfall_df.columns if col.startswith('WW')]
    snapshot_cols.sort(key=lambda x: int(x.replace('WW', '')))

    results = []
    current_projected_inventory = starting_inventory

    # Check if necessary PO columns exist
    po_cols_check = ['GR WW', 'Order Quantity', 'Purchasing Document']
    print(po_df.columns)
    po_cols_exist = all(col in po_df.columns for col in po_cols_check)
    if not po_cols_exist:
        print(f"Warning: Missing one or more required PO columns ({po_cols_check}). PO details will not be included in flags.")

    for snapshot in snapshot_cols:
        week_num = int(snapshot.replace("WW", ""))

        # --- Get demand and planned supply safely handling NaN ---
        demand_plan = 0
        if latest_demand_plan_row is not None and snapshot in latest_demand_plan_row.index:
            demand_val = latest_demand_plan_row[snapshot]
            if not pd.isna(demand_val):
                demand_plan = int(demand_val)

        supply_plan_waterfall = 0
        if latest_supply_plan_row is not None and snapshot in latest_supply_plan_row.index:
             supply_val = latest_supply_plan_row[snapshot]
             if not pd.isna(supply_val):
                  supply_plan_waterfall = int(supply_val)
        # --- End safe handling ---

        # --- Get confirmed PO quantity and list of POs for this week ---
        confirmed_po_qty = 0
        pos_in_week = []
        pos_info = "None"
        if po_cols_exist:
            weekly_pos = po_df[po_df['GR WW'] == week_num]
            if not weekly_pos.empty:
                po_qty_sum = weekly_pos['Order Quantity'].sum()
                if not pd.isna(po_qty_sum):
                    confirmed_po_qty = int(po_qty_sum)

                # Get the list of PO numbers, ensuring they are not NaN/None
                pos_in_week = weekly_pos['Purchasing Document'].dropna().astype(str).tolist()
                # Add quantities to the list for more detail
                pos_with_qty = [f"{doc} ({qty})" for doc, qty in zip(weekly_pos['Purchasing Document'].dropna().astype(str), weekly_pos['Order Quantity'].loc[weekly_pos['Purchasing Document'].dropna().index])]
                pos_info = ", ".join(pos_with_qty) if pos_with_qty else "None"

        # --- End getting PO details ---


        # Calculate projected inventory based on confirmed POs
        projected_inventory = current_projected_inventory + confirmed_po_qty - demand_plan

        # Identify potential issues and suggest adjustments
        suggestions = []

        # Flag 1: Discrepancy between planned waterfall supply and confirmed PO quantity
        if confirmed_po_qty != supply_plan_waterfall and (confirmed_po_qty > 0 or supply_plan_waterfall > 0):
             suggestions.append(f"PO vs Waterfall discrepancy: Planned {supply_plan_waterfall}, Confirmed PO {confirmed_po_qty}. POs in week: [{pos_info}]")

        # Flag 2: Projected shortage (inventory goes negative)
        if projected_inventory < 0:
            needed_qty = abs(projected_inventory)
            suggestions.append(f"Projected shortage of {needed_qty} units.")
            # Suggestion includes relevant POs for context
            suggestions.append(f"Suggestion: Consider increasing quantity on POs in or before WW{week_num} ([{pos_info}]) by ~{needed_qty} units total, or expediting later POs.")

        # Flag 3: Potential excess (inventory is building up significantly)
        if confirmed_po_qty > demand_plan and projected_inventory > demand_plan * 2 and current_projected_inventory >= 0:
             excess_this_week = confirmed_po_qty - demand_plan
             suggestions.append(f"Potential for excess inventory. Confirmed PO {confirmed_po_qty} > Demand {demand_plan}. Projected inventory {projected_inventory} is high. POs in week: [{pos_info}]")
             # Suggestion includes relevant POs for context
             suggestions.append(f"Suggestion: Consider decreasing quantity on POs in WW{week_num} ([{pos_info}]) by up to {excess_this_week} units total, or delaying delivery.")

        # Record results
        results.append({
            'Analysis Week': snapshot,
            'Starting Projected Inventory': current_projected_inventory,
            'Demand Plan (Waterfall)': demand_plan,
            'Supply Plan (Waterfall)': supply_plan_waterfall, # Included for comparison
            'Confirmed PO Quantity': confirmed_po_qty,
            'Confirmed POs in Week': pos_info, # New column to list POs clearly
            'End Projected Inventory (using Confirmed PO)': projected_inventory,
            'Flags & Suggestions': ", ".join(suggestions) if suggestions else "OK - Appears Balanced based on confirmed POs"
        })

        # Update for next loop
        current_projected_inventory = projected_inventory

    return pd.DataFrame(results)