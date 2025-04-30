from groq import Groq
import streamlit as st
import pandas as pd

API_KEY = st.secrets["groq"]["API_KEY"]

models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "llama-guard-3-8b",
    "llama3-70b-8192",
    "llama3-8b-8192"
]


def explain_box_plot_with_groq_consumption(df, material_column="Material Number"):
    """
    Explains boxplot or variance of a DataFrame column using Groq LLM.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to explain.
        api_key (str): Your Groq API key.
    """

    # Calculate statistics using groupby and agg
    material_stats = df.groupby(material_column)['Quantity'].agg(
        min='min',
        max='max',
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        percentile_90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Calculate outliers (IQR method)
    def calculate_outliers(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return list(group[(group < lower_bound) | (group > upper_bound)])

    outliers = df.groupby(material_column)['Quantity'].apply(calculate_outliers).reset_index(name='outliers')

    # Merge outliers with stats
    material_stats = pd.merge(material_stats, outliers, on=material_column)

    # Create a string representation of the statistics
    stats_string = material_stats.to_string(index=False)

    client = Groq(api_key=API_KEY)

    data_description = f"Boxplot data from materials. Quantity data statistics:\n{stats_string}"

    system_prompt = """
    You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
    Your task is to interpret a string description of boxplot statistics representing material quantities at the material number level which are from past historical consumption.
    Provide key insights and interpretations of the boxplot data in bullet points.
    Focus on:
    - Compare and contrast the distribution of 'Quantity' across different material numbers.
    - Highlight significant differences in variance (spread) and skewness.
    - Identify and interpret the impact of notable outliers.
    - Summarize key trends and patterns observed in the data.
    - Integrate into your analysis the understanding that while the box plots are sorted by largest variance, visual inspection of the box plot's spread, interquartile range, or absolute differences between medians can be misleading when trying to directly infer variance. Variance is a statistical measure of the average squared deviation from the mean, and should be calculated directly from the data. Do not simply state this explanation as a separate point; weave it into your analysis.
    - Do not simply repeat the statistical values. Provide analytical interpretations.
    Do not include any introductory phrases or preambles. Start directly with the bullet points.
    """

    user_prompt = f"""
    Explain the boxplot for the following data:\n\n{data_description}
    """

    for model in models:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
            )
            explanation = chat_completion.choices[0].message.content
            st.write(explanation)
            break  # Success, exit loop
        except Exception as e:
            # st.warning(f"Model {model} failed: {e}")
            continue
    else:
        st.error("All model attempts failed.")

def explain_box_plot_with_groq_orderplacement(df, material_column="Material Number"):
    """
    Explains boxplot or variance of a DataFrame column using Groq LLM.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to explain.
        api_key (str): Your Groq API key.
    """

    # Calculate statistics using groupby and agg
    material_stats = df.groupby(material_column)['Order Quantity'].agg(
        min='min',
        max='max',
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        percentile_90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Calculate outliers (IQR method)
    def calculate_outliers(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return list(group[(group < lower_bound) | (group > upper_bound)])

    outliers = df.groupby(material_column)['Order Quantity'].apply(calculate_outliers).reset_index(name='outliers')

    # Merge outliers with stats
    material_stats = pd.merge(material_stats, outliers, on=material_column)

    # Create a string representation of the statistics
    stats_string = material_stats.to_string(index=False)

    client = Groq(api_key=API_KEY)

    data_description = f"Boxplot data from materials. Quantity data statistics:\n{stats_string}"

    system_prompt = """
    You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
    Your task is to interpret a string description of boxplot statistics representing material quantities at the material number level which are from past historical order placement.
    Provide key insights and interpretations of the boxplot data in bullet points.
    Focus on:
    - Compare and contrast the distribution of 'Order Quantity' across different material numbers.
    - Highlight significant differences in variance (spread) and skewness.
    - Identify and interpret the impact of notable outliers.
    - Summarize key trends and patterns observed in the data.
    - Integrate into your analysis the understanding that while the box plots are sorted by largest variance, visual inspection of the box plot's spread, interquartile range, or absolute differences between medians can be misleading when trying to directly infer variance. Variance is a statistical measure of the average squared deviation from the mean, and should be calculated directly from the data. Do not simply state this explanation as a separate point; weave it into your analysis.
    - Do not simply repeat the statistical values. Provide analytical interpretations.
    Do not include any introductory phrases or preambles. Start directly with the bullet points.
    """

    user_prompt = f"""
    Explain the boxplot for the following data:\n\n{data_description}
    """

    for model in models:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
            )
            explanation = chat_completion.choices[0].message.content
            st.write(explanation)
            break  # Success, exit loop
        except Exception as e:
            # st.warning(f"Model {model} failed: {e}")
            continue
    else:
        st.error("All model attempts failed.")

def explain_box_plot_with_groq_goods_receipt(df, material_column="Material Number"):
    """
    Explains boxplot or variance of a DataFrame column using Groq LLM.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to explain.
        api_key (str): Your Groq API key.
    """

    # Calculate statistics using groupby and agg
    material_stats = df.groupby(material_column)['Quantity'].agg(
        min='min',
        max='max',
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        percentile_90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Calculate outliers (IQR method)
    def calculate_outliers(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return list(group[(group < lower_bound) | (group > upper_bound)])

    outliers = df.groupby(material_column)['Quantity'].apply(calculate_outliers).reset_index(name='outliers')

    # Merge outliers with stats
    material_stats = pd.merge(material_stats, outliers, on=material_column)

    # Create a string representation of the statistics
    stats_string = material_stats.to_string(index=False)

    client = Groq(api_key=API_KEY)

    data_description = f"Boxplot data from materials. Quantity data statistics:\n{stats_string}"

    system_prompt = """
    You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
    Your task is to interpret a string description of boxplot statistics representing material quantities at the material number level which are from past historical goods receipt data.
    Provide key insights and interpretations of the boxplot data in bullet points.
    Focus on:
    - Compare and contrast the distribution of 'Quantity' across different material numbers.
    - Highlight significant differences in variance (spread) and skewness.
    - Identify and interpret the impact of notable outliers.
    - Summarize key trends and patterns observed in the data.
    - Integrate into your analysis the understanding that while the box plots are sorted by largest variance, visual inspection of the box plot's spread, interquartile range, or absolute differences between medians can be misleading when trying to directly infer variance. Variance is a statistical measure of the average squared deviation from the mean, and should be calculated directly from the data. Do not simply state this explanation as a separate point; weave it into your analysis.
    - Do not simply repeat the statistical values. Provide analytical interpretations.
    Do not include any introductory phrases or preambles. Start directly with the bullet points.
    """

    user_prompt = f"""
    Explain the boxplot for the following data:\n\n{data_description}
    """

    for model in models:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
            )
            explanation = chat_completion.choices[0].message.content
            st.write(explanation)
            break  # Success, exit loop
        except Exception as e:
            # st.warning(f"Model {model} failed: {e}")
            continue
    else:
        st.error("All model attempts failed.")

# def explain_waterfall_chart_with_groq(df):
#     """
#     Explains the root cause analysis of a waterfall chart.

#     Args:
#         df (pd.DataFrame): The DataFrame containing the data.
#         material_column (str): The name of the column containing material numbers.
#     """
#     df_string = df.to_string(index=False)
#     client = Groq(api_key=API_KEY)

#     system_prompt = """
#             You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
            
#             Your task is to interpret a string description of waterfall chart dataframe representing supply chain values at the material number level which are from past historical data. The data is structured in a weekly format labelled as “WW” and the respective week number.
            
#             Provide key insights and interpretations of the waterfall chart data in bullet points.
            
#             Focus on:
#             1.  Identify any negative values in the rows 'Weeks of Stock', and highlight if there are significant differences of more than 4 weeks. 
#             2.  Identify any major differences between the row 'Demand w/o Buffer' under the 'Measures' row for the material number across the different weeks in the first column.
#             3.  Distinguish between the effects of demand variability and supply-side delays or gaps.
#             4.  If there are any negative values for the 'EOH w/o Buffer', validate within the same week if there are any supply issues or negative inventory or demand requirements.
#             5.  Provide contextual interpretations—do not just state values.
#             6.  Provide recommendations focusing on improving the overall supply to meet demand fluctuations.
#             7.  Infer root causes for observed issues, particularly:
#             * Why does the inventory on hand - 'EOH w/o Buffer' - turn negative and remains negative (e.g., zero supply, poor planning, or missed lead times, unexpected demand)
            
#             Integrate into your analysis the understanding that while the waterfall chart visually represents the cumulative effect of sequential changes, the actual root causes should be inferred from the underlying data and context. Do not simply state this explanation as a separate point; weave it into your analysis.
#             Do not give me generic explanations. Everything has to be backed with the data you have seen. No generic ifs and hows.
            
#             Important: Use the sequential weekly data to derive temporal insights. Understand that the waterfall-style cumulative effect seen in EOH charts reflects decisions made in earlier weeks. Your analysis must go beyond visualization and into operational logic.

#             Do not include any introductory phrases or preambles. Start directly with the bullet points.
#             """

#     user_prompt = f"""
#         Explain the root cause analysis for the following waterfall chart data:\n\n{df_string}
#         """

#     try:
#         chat_completion = client.chat.completions.create(
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             model="llama-3.3-70b-versatile",
#         )

#         explanation = chat_completion.choices[0].message.content
#         st.header("Root Cause Analysis")
#         st.write(explanation)

#     except Exception as e:
#         st.error(f"Error during Groq API call: {e}")

def explain_scenario_4_with_groq(df):
    df_string = df.to_string(index=False)
    client = Groq(api_key=API_KEY)

    system_prompt = """
    
    You are a highly skilled supply chain analyst specializing in the semiconductor industry, with deep experience in analyzing weekly historical data at the material number level. 
    
    You are presented with a text-based description of a weekly snapshot dataframe that includes columns such as Snapshot (labelled by week numbers, e.g., WW08), LeadTime(Week), Changed, and Change Details. 
    
    Your role is to evaluate the trends and anomalies in lead time data.
    
    Perform the following tasks:

    * Assess whether the lead time is longer than usual based on the weekly trend.

    * If there is any change in the lead time, analyze and explain it using the "Changed" and "Change Details" columns.

    * If there is no change, confirm the stability and consistency of the lead time.

    * Deliver a clear, concise analysis in plain language that can be easily shared with both technical and business stakeholders.

    Do not include any introductory phrases or preambles. Start directly with bullet points.
    """

    user_prompt = f"""
        Analyse the following weekly snapshot dataframe:\n\n{df_string}
        """
        
    for model in models:
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=model,
            )
            explanation = chat_completion.choices[0].message.content
            st.write(explanation)
            break  # Success, exit loop
        except Exception as e:
            # st.warning(f"Model {model} failed: {e}")
            continue
    else:
        st.error("All model attempts failed.")

def explain_waterfall_chart_with_groq(df):
    """
    Explains the root cause analysis of a waterfall chart, with chunking for large inputs.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    """
    df_string = df.to_string(index=False)
    client = Groq(api_key=API_KEY)

    system_prompt = """
    
                You are a supply chain analyst with expertise in the semiconductor industry and deep experience in interpreting weekly historical data at the material number level. Your task is to analyze a text-based description of a waterfall chart dataframe representing supply chain metrics, structured weekly (labelled “WW” with week numbers).

                Your analysis must yield bullet-point insights followed by a single root cause conclusion. Focus on:

                * Weeks of Stock: Identify any negative values. Highlight if the drop exceeds 4 weeks.

                * Demand w/o Buffer: Detect major changes across weeks. Note inconsistencies or spikes.

                * EOH w/o Buffer: Flag negative values. Cross-check same week for issues in supply, inventory, or demand.

                * Buffered Demand Changes: Note sharp increases or decreases week-to-week using "Demand with Buffer". Relate these to inventory impacts.

                Avoid generic commentary. All insights must be directly supported by data. Incorporate temporal context: each week’s situation reflects prior weeks’ decisions and constraints.

                Output Format:

                * Bullet points with detailed observations.

                * Root Cause (one of the following scenarios):
                    
                    - Buffered Demand Changes

                Do not provide introductions, summaries, or explanations beyond this format.
    """

    # system_prompt = """
    #     You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
        
    #     Your task is to interpret a string description of waterfall chart dataframe representing supply chain values at the material number level which are from past historical data. The data is structured in a weekly format labelled as “WW” and the respective week number.
        
    #     Provide key insights and interpretations of the waterfall chart data in bullet points.
        
    #     Focus on:
    #     1.  Identify any negative values in the rows 'Weeks of Stock', and highlight if there are significant differences of more than 4 weeks. 
    #     2.  Identify any major differences between the row 'Demand w/o Buffer' under the 'Measures' row for the material number across the different weeks in the first column.
    #     3.  Distinguish between the effects of demand variability and supply-side delays or gaps.
    #     4.  If there are any negative values for the 'EOH w/o Buffer', validate within the same week if there are any supply issues or negative inventory or demand requirements.
    #     5.  Provide contextual interpretations—do not just state values.
    #     6.  Infer root causes for observed issues, particularly:
    #         * Persistent Negative Inventory: Identify where the 'EOH w/o Buffer' remains negative for multiple consecutive weeks. Infer potential root causes for this persistent shortage, considering factors like continuous zero supply, consistently underestimated demand, or unresolved supply chain disruptions.
    #         * Changes in Buffered Demand: Identify any notable increases or decreases in 'Demand with Buffer' from one week to the next, and consider how these changes might relate to observed inventory levels. 
        
    #     Integrate into your analysis the understanding that while the waterfall chart visually represents the cumulative effect of sequential changes, the actual root causes should be inferred from the underlying data and context. Do not simply state this explanation as a separate point; weave it into your analysis.
    #     Do not give generic explanations. Everything must be backed with the provided data.
        
    #     Important: Use the sequential weekly data to derive temporal insights. Understand that the waterfall-style cumulative effect seen in EOH charts reflects decisions made in earlier weeks. Your analysis must go beyond visualization and into operational logic.
        
    #     Do not include any introductory phrases or preambles. Start directly with bullet points.
    # """

    def process_chunk(chunk_text):
            user_prompt = f"Explain the root cause analysis for the following waterfall chart data:\n\n{chunk_text}"
            for model in models:
                try:
                    client = Groq(api_key=API_KEY)
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        model=model,
                    )
                    return chat_completion.choices[0].message.content
                except Exception as e:
                    # st.warning(f"Model {model} failed: {e}")
                    continue
            return "All model attempts failed."

    try:
        explanation = process_chunk(df_string)
        if "All model attempts failed." not in explanation:
            st.header("Root Cause Analysis")
            st.write(explanation)
        else:
            raise Exception("Large input or model limits reached. Chunking required.")
    except Exception:
        # st.warning("Large input detected or failure occurred. Chunking the data for processing...")

        max_divisor = 10
        df_rows = df.shape[0]
        chunk_results = []

        for divisor in range(2, max_divisor + 1):
            chunk_size = df_rows // divisor
            chunks = [df.iloc[i:i + chunk_size] for i in range(0, df_rows, chunk_size)]
            chunk_results.clear()
            chunk_failed = False

            for chunk in chunks:
                chunk_text = chunk.to_string(index=False)
                chunk_result = process_chunk(chunk_text)
                if "All model attempts failed." in chunk_result:
                    chunk_failed = True
                    break
                else:
                    chunk_results.append(chunk_result)

            if not chunk_failed:
                break  # All chunks succeeded

        if chunk_results:
            combined_insights = "\n".join(chunk_results)

            final_prompt = f"""
            You are a supply chain analyst. Consolidate the following root cause analyses into a clean, non-redundant summary.
            Ensure:
            - Insights are merged logically.
            - Redundant bullet points are combined.
            - Only one root cause conclusion is stated at the end.
            - Follow the original format: bullet points followed by a single root cause.

            Data Insights:
            {combined_insights}
            """

            for model in models:
                try:
                    client = Groq(api_key=API_KEY)
                    final_summary = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a senior supply chain analyst. Output clear, concise insights as requested."},
                            {"role": "user", "content": final_prompt},
                        ],
                        model=model,
                    ).choices[0].message.content
                    st.header("Root Cause Analysis (Final Summary)")
                    st.write(final_summary)
                    break
                except Exception as e:
                    # st.warning(f"Final consolidation model {model} failed: {e}")
            else:
                st.error("All model attempts failed for final consolidation.")
        else:
            st.error("Failed to process the data even after chunking.")



def explain_inventory_events(representative_weekly_events, reorder_point, lead_time, lead_time_std_dev, consumption_distribution_params, consumption_type, consumption_best_distribution, order_distribution_params, order_quantity_type, order_distribution_best):
    """
    Explains weekly inventory events in terms of recommended inventory policies to prevent stockout.

    Args:
        representative_weekly_events (list): A list of strings representing the weekly events.
        reorder_point (int): The reorder point used in the simulation.
        lead_time (int): The average lead time used in the simulation.
        lead_time_std_dev (float): The standard deviation of the lead time.
        consumption_distribution_params (dict): Parameters of the consumption distribution, if applicable.
        consumption_type (str): "Fixed" or "Distribution".
        order_distribution_params (dict): Parameters of the order quantity distribution, if applicable.
        order_quantity_type (str): "Fixed" or "Distribution".
    """

    system_prompt = f"""
    You are an expert supply chain analyst specializing in inventory management with extensive experience in analyzing inventory events and recommending policies to prevent stockout. Your role is to analyze the weekly inventory events and provide actionable insights based on your findings.
    Your task is to interpret the weekly inventory events and recommend inventory policies to prevent stockout. 
     Analyze the provided weekly inventory events, considering:
        * Identify Stockout/Near-Stockout Events:
            * Analyze the provided weekly inventory events to pinpoint specific weeks where stockout or near-stockout situations occurred, considering both reactive and proactive inventory levels.
            * Specifically, examine weeks where ending inventory levels dropped significantly or approached zero.
            * Use the provided data to quantify the severity of these events.
            * Proactively suggest potential root causes for these events, based on the provided data. Explain the root cause in detail using the data provided.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Analyze Consumption Patterns:
            * Detail and explain the consumption patterns in the identified critical weeks.
            * Correlate consumption with the provided distribution source (probability distribution identified) and assess if there are any significant deviations from the expected behavior.
            * Calculate and present the variance in consumption week to week.
            * Quantify the difference between forecasted and actual consumption. Actual consumption is the simulated consumption based on the fixed or distribution identified and forecasted consumption is using the forecasting models. 
            * Consider that consumption type is: {consumption_type}. If consumption type is Distribution, the distribution parameters were: {consumption_distribution_params} and the probability distribution it follows is {consumption_best_distribution}. Explain how the distribution parameters affected the consumption, if applicable.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Evaluate Inventory Levels:
            * Compare reactive and proactive inventory levels in the critical weeks.
            * Determine the impact of reactive and proactive orders on preventing or mitigating stockouts.
            * Calculate the time between order and arrival for both reactive and proactive orders.
            * Consider the lead time was {lead_time} with a standard deviation of {lead_time_std_dev}.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Proactive vs. Reactive Inventory:
            * Based on the data, explicitly demonstrate the advantages of proactive inventory ordering over reactive ordering.
            * Quantify the differences in inventory levels and stockout occurrences between the two strategies.
            * Explain how the provided forecasted consumption data could have been better used.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Inventory Policy Recommendations:
            * Provide specific, data-driven recommendations for inventory policies to prevent future stockouts for the given sequence of events.
            * Suggest optimal reorder points, safety stock levels, and order quantities for both reactive and proactive inventory for the given data or how the order placement should have been done better.
            * Recommend adjustments to the forecasting method based on the observed consumption patterns.
            * Recommend a policy that accounts for the variation in the consumption.
            * Recommend a value in which initial inventory should have started off to prevent stockouts. Reason the math mildly.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Ordering Strategy Improvements:
            * Analyze the current ordering strategy and identify its weaknesses.
            * Suggest improvements to the timing and frequency of orders, considering the lead times and consumption variability.
            * Recommend if the reorder point should be changed, and by how much. The reorder point used was {reorder_point}.
            * Consider that order quantity type is: {order_quantity_type}. If order quantity type is Distribution, the distribution parameters were: {order_distribution_params} and the distribution it followed is {order_distribution_best}. Explain how the order distribution parameters affected the order quantity, if applicable.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Summary of Key Insights and Recommendations:
            * Provide a concise summary of the key insights derived from the data.
            * Present a clear and actionable list of recommendations to optimize inventory management and prevent future stockouts.
            * Include the calculated variance of consumption, and the calculated lead times, and the calculated difference between forecasted and actual consumption in the summary.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
    Do not include any introductory phrases or preambles. 
    Each section should have multiple bullet points, and each bullet point may contain more than one sentence where necessary to ensure clarity and depth in the analysis.
    Use the data provided to pinpoint specific weeks and events that led to stockout. Provide detailed, data-driven insights.
    """

    def process_chunk(chunk):
        weekly_events_text = "\n\n".join(chunk)
        user_prompt = f"Explain the weekly inventory events and recommend inventory policies to prevent stockout or excess inventory:\n\n{weekly_events_text}"

        for model in models:
            try:
                client = Groq(api_key=API_KEY)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=model,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                if hasattr(e, "status_code") and e.status_code == 429:
                    #st.warning(f"Rate limit hit on model {model}. Trying next model...")
                    continue  # Try next model
                else:
                    raise e  # Trigger chunking if it's any other error

        return "All model attempts failed."

    try:
        # Try full message first
        result = process_chunk(representative_weekly_events)
        if "All model attempts failed." not in result:
            # st.write(result)
            return
        else:
            raise Exception("Model fallback failed.")

    except Exception as main_error:
        # Fallback to chunking
        # st.warning("Falling back to chunking due to input size or model error.")

        max_divisor = 50
        successful = False
        results = []

        for divisor in range(2, max_divisor + 1):
            chunk_size = len(representative_weekly_events) // divisor
            chunks = [representative_weekly_events[i:i + chunk_size]
                      for i in range(0, len(representative_weekly_events), chunk_size)]

            results.clear()
            chunk_failed = False

            for chunk in chunks:
                try:
                    result = process_chunk(chunk)
                    if "All model attempts failed." in result:
                        chunk_failed = True
                        break
                    else:
                        results.append(result)
                except Exception as e:
                    chunk_failed = True
                    break

            if not chunk_failed:
                successful = True
                break

        if not successful:
            st.error("Failed to process the data even after chunking.")
            return

        # Consolidate analysis
        consolidated_analysis = {
            "Stockout/Near-Stockout Events": [],
            "Consumption Patterns": [],
            "Inventory Levels": [],
            "Proactive vs. Reactive Inventory": [],
            "Inventory Policy Recommendations": [],
            "Ordering Strategy Improvements": [],
            "Summary of Key Insights and Recommendations": []
        }

        for result in results:
            sections = result.split("\n\n")
            for section in sections:
                for key in consolidated_analysis:
                    if key in section:
                        consolidated_analysis[key].append(section)
                        break

        final_report = ""
        for section, content in consolidated_analysis.items():
            final_report += f"{section}:\n"
            for item in content:
                final_report += f"{item}\n\n"

        st.write(final_report)