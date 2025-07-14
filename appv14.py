import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from datetime import datetime, timedelta
import anthropic
import re
import json
import os
import numpy as np
from typing import List, Dict, Optional

# Configure Streamlit
st.set_page_config(
    page_title="SupaBot BI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Training System from v1justsupabot.py
class EnhancedTrainingSystem:
    def __init__(self, training_file="supabot_training.json"):
        self.training_file = training_file
        self.training_data = self.load_training_data()

    def load_training_data(self) -> List[Dict]:
        """Load training examples from JSON file"""
        if os.path.exists(self.training_file):
            try:
                with open(self.training_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_training_data(self):
        """Save training examples to JSON file"""
        try:
            with open(self.training_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to save training data: {e}")
            return False

    def add_training_example(self, question: str, sql: str, feedback: str = "correct", explanation: str = ""):
        """Add a new training example with optional explanation"""
        example = {
            "question": question.lower().strip(),
            "sql": sql.strip(),
            "feedback": feedback,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        self.training_data.append(example)
        return self.save_training_data()

    def find_similar_examples(self, question: str, limit: int = 3) -> List[Dict]:
        """Find similar training examples using enhanced similarity"""
        question = question.lower().strip()
        scored_examples = []
        
        business_terms = {
            'sales': ['revenue', 'income', 'earnings', 'total'],
            'hour': ['time', 'hourly', 'per hour'],
            'store': ['location', 'branch', 'shop'],
            'total': ['sum', 'aggregate', 'combined', 'all'],
            'date': ['day', 'daily', 'time period']
        }
        
        for example in self.training_data:
            if example["feedback"] in ["correct", "corrected"]:
                q1_words = set(question.split())
                q2_words = set(example["question"].split())
                
                if len(q1_words | q2_words) > 0:
                    basic_similarity = len(q1_words & q2_words) / len(q1_words | q2_words)
                    
                    business_score = 0
                    for term, synonyms in business_terms.items():
                        if any(syn in question for syn in [term] + synonyms):
                            if any(syn in example["question"] for syn in [term] + synonyms):
                                business_score += 0.3
                    
                    final_score = basic_similarity + business_score
                    scored_examples.append((final_score, example))
        
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for score, example in scored_examples[:limit] if score > 0.2]

    def get_training_context(self, question: str) -> str:
        """Get relevant training examples formatted as context"""
        similar_examples = self.find_similar_examples(question)
        if not similar_examples:
            return ""
        
        context = "RELEVANT TRAINING EXAMPLES:\n\n"
        for i, example in enumerate(similar_examples, 1):
            context += f"Example {i}:\n"
            context += f"Question: {example['question']}\n"
            context += f"SQL: {example['sql']}\n"
            if example.get('explanation'):
                context += f"Note: {example['explanation']}\n"
            context += "\n"
        
        return context

def get_training_system():
    """Initialize training system with default examples"""
    training_system = EnhancedTrainingSystem()
    
    if len(training_system.training_data) == 0:
        default_examples = [
            {
                "question": "sales per hour total of all stores and all dates",
                "sql": """
                WITH hourly_sales AS (
                    SELECT 
                        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
                        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
                        SUM(ti.item_total) as total_sales
                    FROM transactions t
                    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
                    WHERE LOWER(t.transaction_type) = 'sale' 
                    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
                    GROUP BY EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'), TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM')
                    ORDER BY hour
                )
                SELECT hour, hour_label, COALESCE(total_sales, 0) as total_sales FROM hourly_sales;
                """,
                "feedback": "correct",
                "explanation": "Groups by hour only across ALL stores and dates. Different from per-store breakdown.",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        for example in default_examples:
            training_system.training_data.append(example)
        training_system.save_training_data()
    
    return training_system

def create_db_connection():
    try:
        # Try multiple possible secret configurations
        if "postgres" in st.secrets:
            # Format 1: [postgres] section
            return psycopg2.connect(
                host=st.secrets["postgres"]["host"],
                database=st.secrets["postgres"]["database"],
                user=st.secrets["postgres"]["user"],
                password=st.secrets["postgres"]["password"],
                port=st.secrets["postgres"]["port"]
            )
        else:
            # Format 2: Individual keys (fallback)
            return psycopg2.connect(
                host=st.secrets.get("SUPABASE_HOST", st.secrets.get("host")),
                database=st.secrets.get("SUPABASE_DB", st.secrets.get("database")),
                user=st.secrets.get("SUPABASE_USER", st.secrets.get("user")),
                password=st.secrets.get("SUPABASE_PASSWORD", st.secrets.get("password")),
                port=st.secrets.get("SUPABASE_PORT", st.secrets.get("port", "5432"))
            )
    except KeyError as e:
        st.error(f"Missing database credential: {e}")
        st.info("Please add your database credentials to .streamlit/secrets.toml")
        return None
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

@st.cache_data(ttl=3600)
def get_database_schema():
    """Fetch the complete database schema including sample data"""
    conn = create_db_connection()
    if not conn:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
        tables = cursor.fetchall()
        schema_info = {}
        for (table_name,) in tables:
            cursor.execute(f"SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name = '{table_name}' ORDER BY ordinal_position")
            columns = cursor.fetchall()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_data = cursor.fetchall()
            schema_info[table_name] = {'columns': columns, 'row_count': row_count, 'sample_data': sample_data}
        return schema_info
    except Exception as e:
        st.error(f"Schema fetch failed: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_claude_client():
    try:
        # Try multiple possible secret configurations
        if "anthropic" in st.secrets:
            api_key = st.secrets["anthropic"]["api_key"]
        else:
            # Format 2: Individual keys (fallback)
            api_key = st.secrets.get("CLAUDE_API_KEY", st.secrets.get("ANTHROPIC_API_KEY"))
        
        if api_key:
            return anthropic.Anthropic(api_key=api_key)
        return None
    except:
        return None

# AI Assistant Core Functions
def generate_smart_sql(question, schema_info=None, training_system=None):
    """Ultimate AI SQL generator with training system integration"""
    client = get_claude_client()
    if not client: return None
    
    schema_context = "DATABASE SCHEMA:\n\n"
    if schema_info:
        for table_name, info in schema_info.items():
            schema_context += f"TABLE: {table_name} ({info['row_count']} rows)\nColumns:\n"
            for col_name, data_type, nullable, default in info['columns']:
                nullable_str = "NULL" if nullable == 'YES' else "NOT NULL"
                schema_context += f"  - {col_name}: {data_type} {nullable_str}\n"
            schema_context += "\n"

    training_context = training_system.get_training_context(question) if training_system else ""

    prompt = f"""{schema_context}{training_context}

BUSINESS CONTEXT:
- This is a retail business database tracking sales, inventory, products, and stores.
- Valid sales transactions have: transaction_type = 'sale' AND (is_cancelled IS NULL OR is_cancelled = false).
- TIMEZONE: Data is in Philippines timezone (UTC+8)
- TIME FORMAT: Always format time as 12-hour format (1:00 PM, 7:00 PM, etc.)

CRITICAL AGGREGATION RULES:
1. When user asks for "total across all stores" or "total of all stores" - GROUP BY time/category ONLY, do NOT group by store
2. When user asks for "per store" or "by store" - GROUP BY both store AND time/category
3. Pay attention to the level of aggregation requested.

USER QUESTION: {question}

INSTRUCTIONS:
1. Generate a PostgreSQL query that matches the EXACT aggregation level requested.
2. Use the training examples as reference.
3. For "total across all stores and all dates by hour": GROUP BY hour only.
4. For "sales per store per hour": GROUP BY store AND hour.
5. Use CTEs for readability. Use COALESCE for NULLs. Include meaningful aliases.
6. For time-based queries use AT TIME ZONE 'Asia/Manila' and format time as 12-hour with AM/PM.
7. Order results descending by the main metric.

Generate ONLY the SQL query, no explanations:"""

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        sql = response.content[0].text.strip()
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql).strip()
        if not sql.endswith(';'): sql += ';'
        return sql
    except Exception as e:
        st.error(f"AI query generation failed: {e}")
        return None

def interpret_results(question, results_df, sql_query):
    client = get_claude_client()
    if not client or results_df.empty: return "The query returned no results."
    
    results_summary = f"Query returned {len(results_df)} rows. Columns: {', '.join(results_df.columns)}\n\n"
    results_summary += "First 10 rows:\n" + results_df.head(10).to_string()
    
    prompt = f"""You are a business intelligence expert. The user asked: "{question}"

SQL Query executed:
{sql_query}

Results:
{results_summary}

Please provide a clear, concise, conversational but professional answer to the user's question, followed by key insights and actionable recommendations. Use bullet points. Interpret the data, don't just repeat it. Format monetary amounts as ₱X,XXX (no decimals)."""
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception:
        return "Could not interpret results."

# ENHANCED SMART VISUALIZATION with 8+ Chart Types
def create_smart_visualization(results_df, question):
    """Enhanced visualization function that automatically selects the best chart type"""
    
    if results_df.empty:
        return None
    
    # Get column types
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    text_cols = results_df.select_dtypes(include=['object']).columns.tolist()
    date_cols = results_df.select_dtypes(include=['datetime64', 'datetime']).columns.tolist()
    
    if not numeric_cols:
        return None
    
    # Clean the question for analysis
    question_lower = question.lower()
    
    # Determine chart type based on question keywords and data structure
    chart_type = determine_chart_type(question_lower, results_df, numeric_cols, text_cols, date_cols)
    
    try:
        fig = None
        
        if chart_type == "pie":
            fig = create_pie_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "treemap":
            fig = create_treemap_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "scatter":
            fig = create_scatter_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "line":
            fig = create_line_chart(results_df, question, numeric_cols, text_cols, date_cols)
        elif chart_type == "heatmap":
            fig = create_heatmap_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "box":
            fig = create_box_chart(results_df, question, numeric_cols, text_cols)
        elif chart_type == "area":
            fig = create_area_chart(results_df, question, numeric_cols, text_cols, date_cols)
        else:  # Default to bar chart
            fig = create_bar_chart(results_df, question, numeric_cols, text_cols)
        
        # Apply consistent styling
        if fig:
            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                height=500,
                title_x=0.5
            )
            
        return fig
        
    except Exception as e:
        # Fallback to bar chart if anything fails
        return create_bar_chart(results_df, question, numeric_cols, text_cols)

def determine_chart_type(question_lower, results_df, numeric_cols, text_cols, date_cols):
    """Intelligently determine the best chart type based on question and data"""
    
    # Keywords for different chart types
    pie_keywords = ['distribution', 'breakdown', 'percentage', 'proportion', 'share', 'composition', 'part of']
    treemap_keywords = ['hierarchy', 'treemap', 'nested', 'structure', 'composition', 'size comparison']
    scatter_keywords = ['correlation', 'relationship', 'vs', 'against', 'compare', 'scatter', 'relationship between']
    line_keywords = ['trend', 'over time', 'timeline', 'progression', 'change', 'growth', 'decline']
    heatmap_keywords = ['pattern', 'heatmap', 'intensity', 'by hour and day', 'activity']
    box_keywords = ['outlier', 'distribution', 'quartile', 'median', 'range', 'variance']
    area_keywords = ['cumulative', 'stacked', 'total over time', 'accumulated']
    
    # Check for specific chart type keywords
    if any(keyword in question_lower for keyword in pie_keywords) and len(text_cols) >= 1:
        return "pie"
    
    if any(keyword in question_lower for keyword in treemap_keywords) and len(text_cols) >= 1:
        return "treemap"
    
    if any(keyword in question_lower for keyword in scatter_keywords) and len(numeric_cols) >= 2:
        return "scatter"
    
    if any(keyword in question_lower for keyword in line_keywords) and (date_cols or 'hour' in question_lower or 'day' in question_lower):
        return "line"
    
    if any(keyword in question_lower for keyword in heatmap_keywords):
        return "heatmap"
    
    if any(keyword in question_lower for keyword in box_keywords):
        return "box"
    
    if any(keyword in question_lower for keyword in area_keywords) and date_cols:
        return "area"
    
    # Data-driven decisions
    row_count = len(results_df)
    
    # For small datasets with categories, prefer pie charts for distribution questions
    if row_count <= 10 and len(text_cols) >= 1 and ('category' in question_lower or 'type' in question_lower):
        return "pie"
    
    # For datasets with multiple numeric columns, prefer scatter
    if len(numeric_cols) >= 2 and row_count >= 10:
        return "scatter"
    
    # For time-based data, prefer line charts
    if date_cols or any(col for col in results_df.columns if 'hour' in col.lower() or 'time' in col.lower()):
        return "line"
    
    # Default to bar chart
    return "bar"

def get_best_value_column(numeric_cols):
    """Select the best numeric column for values"""
    priority_terms = ['revenue', 'sales', 'total', 'amount', 'value', 'price', 'cost']
    
    for term in priority_terms:
        for col in numeric_cols:
            if term in col.lower():
                return col
    
    return numeric_cols[0]  # Fallback to first numeric column

def get_best_label_column(text_cols):
    """Select the best text column for labels"""
    priority_terms = ['name', 'category', 'type', 'store', 'product']
    
    for term in priority_terms:
        for col in text_cols:
            if term in col.lower() and 'id' not in col.lower():
                return col
    
    # Return first non-ID column
    for col in text_cols:
        if 'id' not in col.lower():
            return col
    
    return text_cols[0]  # Fallback to first text column

def create_pie_chart(results_df, question, numeric_cols, text_cols):
    """Create a pie chart for distribution/breakdown questions"""
    if not text_cols or not numeric_cols:
        return None
    
    # Select best columns
    value_col = get_best_value_column(numeric_cols)
    label_col = get_best_label_column(text_cols)
    
    # Filter and prepare data
    df_clean = results_df[results_df[value_col] > 0].copy()
    if len(df_clean) > 10:  # Limit to top 10 for readability
        df_clean = df_clean.nlargest(10, value_col)
    
    fig = px.pie(df_clean, values=value_col, names=label_col,
                title=f"Distribution: {question}")
    
    return fig

def create_treemap_chart(results_df, question, numeric_cols, text_cols):
    """Create a treemap for hierarchical data"""
    if not text_cols or not numeric_cols:
        return None
    
    value_col = get_best_value_column(numeric_cols)
    label_col = get_best_label_column(text_cols)
    
    df_clean = results_df[results_df[value_col] > 0].copy()
    if len(df_clean) > 20:
        df_clean = df_clean.nlargest(20, value_col)
    
    fig = px.treemap(df_clean, path=[label_col], values=value_col,
                    title=f"Treemap: {question}")
    
    return fig

def create_scatter_chart(results_df, question, numeric_cols, text_cols):
    """Create a scatter plot for correlation analysis"""
    if len(numeric_cols) < 2:
        return None
    
    x_col = numeric_cols[0]
    y_col = numeric_cols[1]
    
    # If there's a third numeric column, use it for size
    size_col = numeric_cols[2] if len(numeric_cols) > 2 else None
    
    # If there's a text column, use it for color
    color_col = text_cols[0] if text_cols else None
    
    fig = px.scatter(results_df, x=x_col, y=y_col,
                    size=size_col, color=color_col,
                    title=f"Relationship: {question}",
                    hover_data=text_cols[:2] if text_cols else None)
    
    return fig

def create_line_chart(results_df, question, numeric_cols, text_cols, date_cols):
    """Create a line chart for time series data"""
    if not numeric_cols:
        return None
    
    # Determine x-axis (time-based)
    x_col = None
    if date_cols:
        x_col = date_cols[0]
    else:
        # Look for time-related columns
        for col in results_df.columns:
            if any(time_word in col.lower() for time_word in ['hour', 'time', 'date', 'day']):
                x_col = col
                break
    
    if not x_col:
        x_col = results_df.columns[0]  # Fallback to first column
    
    y_col = get_best_value_column(numeric_cols)
    
    # Sort by x-axis for proper line connection
    df_sorted = results_df.sort_values(x_col)
    
    fig = px.line(df_sorted, x=x_col, y=y_col,
                 title=f"Trend: {question}",
                 markers=True)
    
    return fig

def create_heatmap_chart(results_df, question, numeric_cols, text_cols):
    """Create a heatmap for pattern analysis"""
    if len(results_df.columns) < 3:
        return None
    
    # Try to create a pivot table for heatmap
    if len(text_cols) >= 2 and len(numeric_cols) >= 1:
        try:
            pivot_df = results_df.pivot_table(
                index=text_cols[0],
                columns=text_cols[1],
                values=numeric_cols[0],
                fill_value=0
            )
            
            fig = px.imshow(pivot_df,
                           title=f"Pattern Analysis: {question}",
                           aspect="auto",
                           color_continuous_scale="Blues")
            
            return fig
        except:
            pass
    
    # Fallback to correlation heatmap if multiple numeric columns
    if len(numeric_cols) >= 3:
        corr_matrix = results_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix,
                       title=f"Correlation: {question}",
                       color_continuous_scale="RdBu_r",
                       aspect="auto")
        return fig
    
    return None

def create_box_chart(results_df, question, numeric_cols, text_cols):
    """Create box plots for distribution analysis"""
    if not numeric_cols:
        return None
    
    y_col = get_best_value_column(numeric_cols)
    x_col = text_cols[0] if text_cols else None
    
    if x_col:
        fig = px.box(results_df, x=x_col, y=y_col,
                    title=f"Distribution: {question}")
    else:
        fig = px.box(results_df, y=y_col,
                    title=f"Distribution: {question}")
    
    return fig

def create_area_chart(results_df, question, numeric_cols, text_cols, date_cols):
    """Create area chart for cumulative data"""
    if not numeric_cols:
        return None
    
    x_col = date_cols[0] if date_cols else results_df.columns[0]
    y_col = get_best_value_column(numeric_cols)
    
    df_sorted = results_df.sort_values(x_col)
    
    fig = px.area(df_sorted, x=x_col, y=y_col,
                 title=f"Cumulative: {question}")
    
    return fig

def create_bar_chart(results_df, question, numeric_cols, text_cols):
    """Create bar chart (original functionality)"""
    if not text_cols or not numeric_cols:
        return None
    
    y_col = get_best_value_column(numeric_cols)
    x_col = get_best_label_column(text_cols)
    
    df_filtered = results_df[results_df[y_col] > 0].copy()
    if df_filtered.empty:
        return None
        
    df_sorted = df_filtered.sort_values(by=y_col, ascending=False).head(25)
    
    # Determine orientation based on label length
    chart_type = 'h' if any(len(str(s)) > 15 for s in df_sorted[x_col]) else 'v'
    
    if chart_type == 'h':
        fig = px.bar(df_sorted, x=y_col, y=x_col, orientation='h',
                    title=f"Analysis: {question}")
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    else:
        fig = px.bar(df_sorted, x=x_col, y=y_col,
                    title=f"Analysis: {question}")
    
    return fig

# Execute query for AI Assistant
def execute_query_for_assistant(sql):
    conn = create_db_connection()
    if not conn: 
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '30s'")
        df = pd.read_sql(sql, conn)
        # Format datetime columns properly
        for col in df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M')
        return df
    except psycopg2.errors.QueryCanceled:
        st.error("Query took too long to execute. Try a simpler question.")
        return None
    except Exception as e:
        error_msg = str(e)
        st.error(f"Query execution failed: {error_msg}")
        if "does not exist" in error_msg: 
            st.info("💡 The query references a table or column that doesn't exist.")
        elif "syntax error" in error_msg: 
            st.info("💡 There's a syntax error in the SQL.")
        return None
    finally:
        if conn: 
            conn.close()

def get_column_config(df):
    """Dynamic formatting for dataframes from v1justsupabot.py"""
    config = {}
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['revenue', 'profit', 'price', 'cost', 'total', 'amount', 'value', 'sales']):
            config[col] = st.column_config.NumberColumn(label=col.replace("_", " ").title(), format="₱%d")
        elif any(keyword in col_lower for keyword in ['quantity', 'count', 'sold', 'items', 'transactions']):
             config[col] = st.column_config.NumberColumn(label=col.replace("_", " ").title(), format="%,d")
        else:
            config[col] = st.column_config.TextColumn(label=col.replace("_", " ").title())
    return config

# Dashboard Data Fetching Functions (from appv8.py)
def execute_query_for_dashboard(sql):
    conn = create_db_connection()
    if not conn: 
        return None
    try:
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '30s'")
        df = pd.read_sql(sql, conn)
        return df
    except Exception as e:
        # Silently handle errors for dashboard queries to avoid breaking the UI
        print(f"Dashboard query error: {e}")
        return None
    finally:
        if conn: 
            conn.close()

@st.cache_data(ttl=300)
def get_latest_metrics():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        COALESCE(SUM(ti.item_total), 0) as latest_sales,
        COUNT(DISTINCT t.ref_id) as latest_transactions
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_previous_metrics():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        COALESCE(SUM(ti.item_total), 0) as previous_sales,
        COUNT(DISTINCT t.ref_id) as previous_transactions
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date - INTERVAL '1 day'
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_hourly_sales():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila') as hour,
        TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM') as hour_label,
        COALESCE(SUM(ti.item_total), 0) as sales
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    GROUP BY EXTRACT(HOUR FROM t.transaction_time AT TIME ZONE 'Asia/Manila'), TO_CHAR((t.transaction_time AT TIME ZONE 'Asia/Manila'), 'HH12:00 AM')
    HAVING SUM(ti.item_total) > 0 ORDER BY hour
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_store_performance():
    """Get store performance for the latest day only (to match hourly sales)"""
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        s.name as store_name, 
        COALESCE(SUM(ti.item_total), 0) as total_sales
    FROM stores s
    LEFT JOIN transactions t ON s.id = t.store_id
    LEFT JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') = ld.max_date
    GROUP BY s.name
    HAVING COALESCE(SUM(ti.item_total), 0) > 0
    ORDER BY total_sales DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_daily_trend(days=30):
    sql = f"""
    SELECT 
        DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        COALESCE(SUM(ti.item_total), 0) as daily_sales
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= CURRENT_DATE - INTERVAL '{days} days'
    GROUP BY DATE(t.transaction_time AT TIME ZONE 'Asia/Manila')
    HAVING SUM(ti.item_total) > 0
    ORDER BY date
    """
    df = execute_query_for_dashboard(sql)
    if df is not None and not df.empty:
        df['cumulative_sales'] = df['daily_sales'].cumsum()
    return df

@st.cache_data(ttl=300)
def get_store_count():
    sql = "SELECT COUNT(DISTINCT id) as store_count FROM stores"
    result = execute_query_for_dashboard(sql)
    return result.iloc[0]['store_count'] if result is not None and len(result) > 0 else 0

@st.cache_data(ttl=300)
def get_product_performance():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        p.name as product_name,
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY p.name
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_transaction_analysis():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        COUNT(ti.id) as items_per_transaction,
        SUM(ti.item_total) as total_value,
        AVG(ti.item_total) as avg_item_value
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY t.ref_id
    HAVING COUNT(ti.id) > 0
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_daily_sales_by_store():
    sql = """
    SELECT 
        DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') as date,
        s.name as store_name,
        COALESCE(SUM(ti.item_total), 0) as daily_sales
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    JOIN stores s ON t.store_id = s.id
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY DATE(t.transaction_time AT TIME ZONE 'Asia/Manila'), s.name
    ORDER BY date DESC
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_transaction_values_by_store():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        s.name as store_name,
        SUM(ti.item_total) as total_value
    FROM transactions t
    JOIN transaction_items ti ON ti.transaction_ref_id = t.ref_id
    JOIN stores s ON t.store_id = s.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY s.name, t.ref_id
    HAVING SUM(ti.item_total) > 0
    """
    return execute_query_for_dashboard(sql)

@st.cache_data(ttl=300)
def get_category_sales():
    sql = """
    WITH latest_date AS (
        SELECT MAX(DATE(transaction_time AT TIME ZONE 'Asia/Manila')) as max_date
        FROM transactions 
        WHERE LOWER(transaction_type) = 'sale' AND (is_cancelled = false OR is_cancelled IS NULL)
    )
    SELECT 
        p.category as product_category,
        SUM(ti.item_total) as total_revenue
    FROM transaction_items ti
    JOIN transactions t ON ti.transaction_ref_id = t.ref_id
    JOIN products p ON ti.product_id = p.id
    CROSS JOIN latest_date ld
    WHERE LOWER(t.transaction_type) = 'sale' 
    AND (t.is_cancelled = false OR t.is_cancelled IS NULL)
    AND DATE(t.transaction_time AT TIME ZONE 'Asia/Manila') >= ld.max_date - INTERVAL '7 days'
    GROUP BY p.category
    HAVING SUM(ti.item_total) > 0
    ORDER BY total_revenue DESC
    """
    return execute_query_for_dashboard(sql)

def create_calendar_heatmap(df_cal, date_col, value_col):
    """Create calendar heatmap visualization"""
    df_cal = df_cal.copy()
    df_cal[date_col] = pd.to_datetime(df_cal[date_col])
    df_cal = df_cal.sort_values(date_col)
    
    # Create week-based calendar
    df_cal['week'] = df_cal[date_col].dt.isocalendar().week
    df_cal['day_of_week'] = df_cal[date_col].dt.dayofweek
    df_cal['day_num'] = df_cal[date_col].dt.day
    
    min_week = df_cal['week'].min()
    df_cal['week_normalized'] = df_cal['week'] - min_week
    
    unique_weeks = sorted(df_cal['week_normalized'].unique())
    if not unique_weeks or len(df_cal) < 7:
        return None  # Return None if insufficient data
    
    matrix = []
    annotations = []
    
    # Build matrix from Monday (0) to Sunday (6)
    for day in range(7):  # 0 = Monday, 6 = Sunday
        row = []
        for week in unique_weeks:
            day_data = df_cal[(df_cal['week_normalized'] == week) & (df_cal['day_of_week'] == day)]
            if not day_data.empty:
                value = day_data.iloc[0][value_col]
                day_num = day_data.iloc[0]['day_num']
                row.append(value)
                text_color = 'black' if value <= max(df_cal[value_col]) * 0.5 else 'white'
                annotations.append(dict(
                    x=week, y=day,  # y-coordinate matches day index (0 = Monday, 6 = Sunday)
                    text=str(day_num),
                    showarrow=False,
                    font=dict(color=text_color, size=12, family='Arial Black'),
                    xref='x', yref='y'
                ))
            else:
                row.append(0)  # No data for this day
        matrix.append(row)
    
    # Create heatmap with UI-matched color scheme
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=[
            [0, '#f0f4f8'],    # Very light gray-blue for no data
            [0.2, '#d9e6f2'],  # Light blue for low sales
            [0.4, '#a3c9e0'],  # Medium blue for medium sales
            [0.6, '#6baed6'],  # Darker blue for high sales
            [0.8, '#3182bd'],  # Deep blue for higher sales
            [1, '#1b4d7e']     # Dark blue for maximum
        ],
        showscale=True,
        colorbar=dict(
            title="Sales (PHP)",
            tickmode="linear",
            tick0=0,
            dtick=max(df_cal[value_col]) / 5 if df_cal[value_col].max() > 0 else 1,
            tickformat=",.0f"
        ),
        hovertemplate='<b>%{text}</b><br>Sales: ₱%{z:,.0f}<extra></extra>',
        text=[[f"{['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]}" for week in range(len(unique_weeks))] for day in range(7)]
    ))
    
    # Add day number annotations
    for ann in annotations:
        fig.add_annotation(ann)
    
    # Update layout with Monday at top, Sunday at bottom
    fig.update_layout(
        title=f'📅 Daily Sales Calendar ({df_cal[date_col].min().strftime("%b %d")} - {df_cal[date_col].max().strftime("%b %d, %Y")})',
        xaxis=dict(
            title="",
            tickvals=list(range(len(unique_weeks))),
            ticktext=[f"Week {i+1}" for i in range(len(unique_weeks))],
            side='top'
        ),
        yaxis=dict(
            title="",
            tickvals=list(range(7)),  # 0 = Monday, 6 = Sunday
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],  # Monday at top
            autorange=True  # Automatically adjust while respecting ticktext order
        ),
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        title_x=0.5,
        font=dict(size=11)
    )
    
    return fig

# CSS Styling
def load_css():
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .main-header {
        background: linear-gradient(90deg #00d2ff 0% #3a47d5 100%); /* Fixed syntax error */
        padding: 1.5rem; border-radius: 10px; text-align: center;
        color: white; margin-bottom: 2rem;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: bold; }
    
    /* KPI Metric Boxes - Ensure Equal Height */
    div[data-testid="stMetric"] {
        background-color: #1c1e26; 
        border: 1px solid #2e303d;
        padding: 1.5rem; 
        border-radius: 10px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    div[data-testid="stMetric"] > div:nth-child(1) {
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stMetric"] > div:nth-child(2) {
        font-size: 2rem; 
        font-weight: bold; 
        color: #00d2ff;
        margin-bottom: 0.3rem;
    }
    div[data-testid="stMetric"] > div:nth-child(3) {
        font-size: 0.8rem;
    }
    
    .insight-box {
        background: #16a085; padding: 1rem; border-radius: 8px;
        color: white; margin-top: 1rem; text-align: center;
    }
    .user-message{
        background:linear-gradient(135deg, #3a47d5 0%, #00d2ff 100%);
        padding:1rem 1.5rem; border-radius:20px 20px 0 20px;
        margin:1rem 0; color:white; font-weight:500;
    }
    .ai-message{
        background: #262730; border: 1px solid #3d3d3d;
        padding:1rem 1.5rem; border-radius:20px 20px 20px 0;
        margin:1rem 0; color:white;
    }
    button[data-baseweb="tab"] {
        background-color: transparent;
        border-bottom: 2px solid transparent;
        font-size: 1.1rem;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #00d2ff;
        color: #00d2ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "training_system" not in st.session_state:
        st.session_state.training_system = get_training_system()
    if "schema_info" not in st.session_state:
        st.session_state.schema_info = None

# Main Dashboard Rendering Function
def render_dashboard():
    st.markdown('<div class="main-header"><h1>📊 SupaBot Ultimate BI Dashboard</h1><p>Real-time Business Intelligence powered by AI</p></div>', unsafe_allow_html=True)
    
    # Data Fetching
    latest_data = get_latest_metrics()
    previous_data = get_previous_metrics()
    store_count = get_store_count()
    
    latest_sales = latest_data.iloc[0]['latest_sales'] if latest_data is not None and len(latest_data) > 0 else 0
    latest_transactions = latest_data.iloc[0]['latest_transactions'] if latest_data is not None and len(latest_data) > 0 else 0
    previous_sales = previous_data.iloc[0]['previous_sales'] if previous_data is not None and len(previous_data) > 0 else 0
    previous_transactions = previous_data.iloc[0]['previous_transactions'] if previous_data is not None and len(previous_data) > 0 else 0
    
    sales_growth = ((latest_sales - previous_sales) / max(previous_sales, 1)) * 100
    trans_growth = ((latest_transactions - previous_transactions) / max(previous_transactions, 1)) * 100
    avg_transaction = latest_sales / max(latest_transactions, 1)
    
    # KPI Section
    st.subheader("🚀 Today's Snapshot")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("💰 Latest Day's Sales", f"₱{latest_sales:,.0f}", f"{sales_growth:+.1f}%")
    
    with kpi2:
        st.metric("🛒 Latest Day's Transactions", f"{latest_transactions:,}", f"{trans_growth:+.1f}%")
    
    with kpi3:
        st.metric("💳 Avg Transaction Value", f"₱{avg_transaction:,.0f}")
    
    with kpi4:
        st.metric("🏪 Active Stores", f"{store_count:,}")
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Tabbed Interface for Basic and Advanced Views
    basic_tab, advanced_tab = st.tabs(["📊 Basic Data", "🚀 Advanced Data"])

    with basic_tab:
        st.header("📈 Sales Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly Sales Chart
            st.markdown("#### 🕒 Sales by Hour (Latest Day)")
            hourly_data = get_hourly_sales()
            if hourly_data is not None and not hourly_data.empty:
                fig_hourly = px.bar(hourly_data, x='hour_label', y='sales', 
                                   title='Sales by Hour (Latest Day)', 
                                   labels={'sales': 'Sales (PHP)', 'hour_label': 'Hour'})
                fig_hourly.update_layout(
                    template="plotly_dark", 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    height=450,
                    title_x=0.5
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.info("No hourly sales data available.")

        with col2:
            # Store Performance Chart - Latest Day Only
            st.markdown("#### 🏪 Top Stores by Sales (Latest Day)")
            store_data = get_store_performance()
            if store_data is not None and not store_data.empty:
                fig_stores = px.bar(store_data.head(10), x='store_name', y='total_sales',
                                   title='Top 10 Stores by Sales (Latest Day)',
                                   labels={'total_sales': 'Total Sales (PHP)', 'store_name': 'Store'})
                fig_stores.update_layout(
                    template="plotly_dark", 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    height=450,
                    title_x=0.5
                )
                fig_stores.update_xaxes(tickangle=45)
                st.plotly_chart(fig_stores, use_container_width=True)
            else:
                st.info("No store performance data available.")
        
        # Daily Trend Chart (full width)
        st.markdown("#### 📊 Daily Sales Trend (Last 30 Days)")
        daily_data = get_daily_trend()
        if daily_data is not None and not daily_data.empty:
            fig_daily = px.line(daily_data, x='date', y='daily_sales',
                               title='Daily Sales Trend (Last 30 Days)',
                               labels={'daily_sales': 'Daily Sales (PHP)', 'date': 'Date'})
            fig_daily.update_layout(
                template="plotly_dark", 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)', 
                height=450,
                title_x=0.5
            )
            st.plotly_chart(fig_daily, use_container_width=True)
            st.markdown('<div class="insight-box">📊 Track your daily performance patterns and identify trends</div>', unsafe_allow_html=True)

    with advanced_tab:
        st.header("🚀 Advanced Analytics")
        
        # 3x3 Grid Layout for Charts
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Product Performance Treemap
            st.markdown("#### 🧩 Product Revenue Treemap (Top 15)")
            product_data = get_product_performance()
            if product_data is not None and not product_data.empty:
                fig_treemap = px.treemap(product_data.head(15), 
                                        path=['product_name'], values='total_revenue',
                                        title='Product Revenue Treemap (Top 15)',
                                        labels={'total_revenue': 'Revenue (PHP)'})
                fig_treemap.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300)
                st.plotly_chart(fig_treemap, use_container_width=True)
                st.markdown('<div class="insight-box">💡 Larger blocks = higher revenue products</div>', unsafe_allow_html=True)
            else:
                st.info("No product performance data available.")

        with col2:
            # Transaction Analysis Scatter Plot
            st.markdown("#### 📈 Items vs Total Value")
            transaction_data = get_transaction_analysis()
            if transaction_data is not None and not transaction_data.empty:
                fig = px.scatter(transaction_data, 
                                x='items_per_transaction', y='total_value',
                                title='Transaction Analysis: Items vs Total Value',
                                labels={'items_per_transaction': 'Number of Items', 'total_value': 'Transaction Value (PHP)'},
                                size='total_value', 
                                color='total_value',
                                color_continuous_scale=[[0, '#f0f4f8'], [0.2, '#d9e6f2'], [0.4, '#a3c9e0'], [0.6, '#6baed6'], [0.8, '#3182bd'], [1, '#1b4d7e']])
                fig.update_traces(marker=dict(size=10, color='#6baed6'))
                fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, title_x=0.5)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('<div class="insight-box">ℹ️ Each dot is a transaction. See if more items mean higher value.</div>', unsafe_allow_html=True)
            else:
                st.info("No transaction analysis data available.")

        with col3:
            # Transaction Value Distribution by Store (Box Plot)
            st.markdown("#### 📦 Transaction Value Distribution")
            box_data = get_transaction_values_by_store()
            if box_data is not None and not box_data.empty:
                fig_box = px.box(box_data, x='store_name', y='total_value',
                                title='Transaction Value Distribution by Store (Last 7 Days)',
                                labels={'total_value': 'Transaction Value (PHP)', 'store_name': 'Store'})
                fig_box.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, title_x=0.5)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No transaction value data available.")

        # Second Row
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Sales by Product Category (Donut Chart)
            st.markdown("#### 🍩 Sales by Product Category")
            category_data = get_category_sales()
            if category_data is not None and not category_data.empty:
                fig_donut = px.pie(category_data, values='total_revenue', names='product_category',
                                  title='Sales by Product Category (Last 7 Days)',
                                  hole=0.3)
                fig_donut.update_traces(marker=dict(colors=['#d9e6f2', '#a3c9e0', '#6baed6', '#3182bd', '#1b4d7e']))
                fig_donut.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, title_x=0.5)
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.info("No category sales data available.")

        with col5:
            # Daily Sales by Store (Stacked Bar Chart)
            st.markdown("#### 📊 Daily Sales by Store")
            store_sales_data = get_daily_sales_by_store()
            if store_sales_data is not None and not store_sales_data.empty:
                fig_stacked = px.bar(store_sales_data, x='date', y='daily_sales', color='store_name',
                                    title='Daily Sales by Store (Last 30 Days)',
                                    labels={'daily_sales': 'Sales (PHP)', 'date': 'Date', 'store_name': 'Store'},
                                    height=300)
                fig_stacked.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', title_x=0.5)
                st.plotly_chart(fig_stacked, use_container_width=True)
            else:
                st.info("No daily sales by store data available.")

        with col6:
            # Cumulative Sales Trend (Area Chart)
            st.markdown("#### 📈 Cumulative Sales Trend")
            daily_data = get_daily_trend()
            if daily_data is not None and not daily_data.empty:
                fig_area = px.area(daily_data, x='date', y='cumulative_sales',
                                  title='Cumulative Sales Trend (Last 30 Days)',
                                  labels={'cumulative_sales': 'Cumulative Sales (PHP)', 'date': 'Date'},
                                  height=300)
                fig_area.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', title_x=0.5)
                st.plotly_chart(fig_area, use_container_width=True)
            else:
                st.info("No cumulative sales data available.")

        # Full-width Charts
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Calendar Heatmap (full width)
        st.markdown("#### 📅 Sales Calendar Heatmap")
        daily_data_cal = get_daily_trend(days=365)
        if daily_data_cal is not None and not daily_data_cal.empty:
            if len(daily_data_cal) >= 7:
                cal_fig = create_calendar_heatmap(daily_data_cal, 'date', 'daily_sales')
                st.plotly_chart(cal_fig, use_container_width=True)
                total_days = len(daily_data_cal)
                date_range = f"{daily_data_cal['date'].min().strftime('%b %d')} - {daily_data_cal['date'].max().strftime('%b %d, %Y')}"
                total_sales = daily_data_cal['daily_sales'].sum()
                st.markdown(f'<div class="insight-box">📊 Showing {total_days} days of sales data ({date_range}) • Total: ₱{total_sales:,.0f}</div>', unsafe_allow_html=True)
            else:
                st.info(f"Calendar view needs at least 7 days of data. Currently have {len(daily_data_cal)} days.")
        else:
            st.info("No daily sales data available for calendar view.")
        
        # Correlation Heatmap (full width)
        st.markdown("#### 🔥 Correlation Heatmap (Transaction Metrics)")
        transaction_data = get_transaction_analysis()
        if transaction_data is not None and not transaction_data.empty:
            corr_matrix = transaction_data[['items_per_transaction', 'total_value', 'avg_item_value']].corr()
            fig_heatmap = px.imshow(corr_matrix,
                                  title='Correlation Heatmap (Last 7 Days)',
                                  color_continuous_scale='Blues',
                                  text_auto=True,
                                  aspect="auto")
            fig_heatmap.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=450, title_x=0.5)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("No correlation data available.")

# Enhanced Chat Page with Assistant
def render_chat():
    st.markdown('<div class="main-header"><h1>🧠 SupaBot AI Assistant</h1><p>Ask ANYTHING about your data - I learn from your feedback!</p></div>', unsafe_allow_html=True)

    if st.session_state.schema_info is None:
        with st.spinner("🔍 Learning about your database..."):
            st.session_state.schema_info = get_database_schema()

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">💭 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            if message.get("interpretation"): 
                st.markdown(f'<div class="ai-message">{message["interpretation"]}</div>', unsafe_allow_html=True)
            
            if message.get("sql"):
                with st.expander("🔍 SQL Query & Training", expanded=False):
                    st.code(message["sql"], language="sql")
                    
                    st.markdown("**Was this SQL correct?**")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    with col1:
                        if st.button("✅ Correct", key=f"correct_{i}"):
                            explanation = st.text_input(
                                "Why was this correct?",
                                placeholder="e.g., Perfect grouping for hourly totals",
                                key=f"correct_explanation_{i}"
                            )
                            if st.session_state.training_system.add_training_example(
                                message.get("question", ""), 
                                message["sql"], 
                                "correct", 
                                explanation
                            ):
                                st.success("✅ Saved as correct example!")
                            else:
                                st.error("❌ Failed to save")
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ Wrong", key=f"wrong_{i}"):
                            st.session_state[f"show_correction_{i}"] = True
                            st.rerun()
                    
                    # Show correction interface
                    if st.session_state.get(f"show_correction_{i}", False):
                        st.markdown("**Provide the correct SQL:**")
                        corrected_sql = st.text_area(
                            "Correct SQL:", 
                            value=message["sql"], 
                            height=100,
                            key=f"corrected_sql_{i}"
                        )
                        explanation = st.text_input(
                            "What was wrong?",
                            placeholder="e.g., Should group by store_id for per-store breakdown",
                            key=f"correction_explanation_{i}"
                        )
                        
                        if st.button("💾 Save Correction", key=f"save_correction_{i}"):
                            if st.session_state.training_system.add_training_example(
                                message.get("question", ""), 
                                corrected_sql, 
                                "corrected", 
                                explanation
                            ):
                                st.success("✅ Correction saved!")
                            else:
                                st.error("❌ Failed to save correction")
                            st.session_state[f"show_correction_{i}"] = False
                            st.rerun()
            
            if message.get("results") is not None:
                results = message["results"]
                if isinstance(results, pd.DataFrame) and not results.empty:
                    # Apply dynamic formatting to the dataframe
                    column_config = get_column_config(results)
                    with st.expander(f"📊 View Data ({len(results)} rows)", expanded=False): 
                        st.dataframe(results, column_config=column_config, use_container_width=True, hide_index=True)
                    if message.get("chart"): 
                        st.plotly_chart(message["chart"], use_container_width=True)
            elif message.get("error"): 
                st.error(message["error"])

    if not st.session_state.messages:
        st.markdown("### 💡 Example Questions You Can Ask:")
        c1, c2 = st.columns(2)
        c1.markdown("**🎯 Enhanced Chart Examples:**")
        c1.markdown("- **Pie Chart**: 'Sales distribution by category'")
        c1.markdown("- **Treemap**: 'Product revenue hierarchy'")
        c1.markdown("- **Scatter Plot**: 'Revenue vs quantity relationship'")
        c1.markdown("- **Line Chart**: 'Sales trend over time'")
        
        c2.markdown("**📊 Business Questions:**")
        c2.markdown("- **Performance**: 'Top 10 products by revenue'")
        c2.markdown("- **Time Analysis**: 'Sales per hour total of all stores'")
        c2.markdown("- **Inventory**: 'Which products are almost out of stock?'")
        c2.markdown("- **Correlation**: 'Is there a relationship between price and sales?'")
        
        # Show training system status
        if len(st.session_state.training_system.training_data) > 0:
            st.info(f"🎓 Training System Active: {len(st.session_state.training_system.training_data)} examples learned")

    # Updated chat input with training system
    if prompt := st.chat_input("Ask me anything about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🧠 Thinking with training data..."):
            # Pass training system to SQL generation
            sql = generate_smart_sql(prompt, st.session_state.schema_info, st.session_state.training_system)
            if sql:
                with st.spinner("📊 Analyzing your data..."):
                    results = execute_query_for_assistant(sql)
                if results is not None:
                    with st.spinner("💡 Generating insights & smart visualization..."):
                        interpretation = interpret_results(prompt, results, sql)
                        chart = create_smart_visualization(results, prompt)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "question": prompt, 
                        "sql": sql, 
                        "results": results, 
                        "interpretation": interpretation, 
                        "chart": chart, 
                        "error": None
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "error": "I couldn't process that question. The query failed."
                    })
            else:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "error": "I couldn't generate a query for that question. Try being more specific."
                })
        st.rerun()

def render_settings():
    st.markdown('<div class="main-header"><h1>⚙️ Settings</h1><p>Manage your dashboard</p></div>', unsafe_allow_html=True)
    
    # Configuration Status
    st.subheader("🔧 Configuration Status")
    
    # Check database connection
    db_conn = create_db_connection()
    if db_conn:
        st.success("✅ Database connection successful")
        db_conn.close()
    else:
        st.error("❌ Database connection failed")
        st.info("Add your database credentials to .streamlit/secrets.toml:")
        st.code("""
[postgres]
host = "your-database-host"
database = "your-database-name"
user = "your-database-user"
password = "your-database-password"
port = "5432"

# OR use individual keys:
# host = "your-database-host"
# database = "your-database-name"
# user = "your-database-user"
# password = "your-database-password"
# port = "5432"
        """)
    
    # Check API key
    claude_client = get_claude_client()
    if claude_client:
        st.success("✅ Claude API key configured")
    else:
        st.error("❌ Claude API key missing")
        st.info("Add your Anthropic API key to .streamlit/secrets.toml:")
        st.code("""
[anthropic]
api_key = "your-anthropic-api-key"

# OR use direct key:
# CLAUDE_API_KEY = "your-anthropic-api-key"
        """)
    
    st.subheader("🎓 Training System")
    training_count = len(st.session_state.training_system.training_data)
    st.metric("Training Examples", training_count)
    if training_count > 0:
        correct_count = len([ex for ex in st.session_state.training_system.training_data if ex["feedback"] == "correct"])
        corrected_count = len([ex for ex in st.session_state.training_system.training_data if ex["feedback"] == "corrected"])
        st.write(f"✅ Correct: {correct_count}")
        st.write(f"🔧 Corrected: {corrected_count}")
        
        with st.expander("📋 View Training Data"):
            for example in st.session_state.training_system.training_data[-5:]:
                st.write(f"**Q:** {example['question']}")
                st.write(f"**Status:** {example['feedback']}")
                if example.get('explanation'):
                    st.write(f"**Note:** {example['explanation']}")
                st.write("---")
    
    st.subheader("🛠️ Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Training Data"):
            st.session_state.training_system.training_data = []
            st.session_state.training_system.save_training_data()
            st.success("Training data cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Refresh Cache"):
            st.cache_data.clear()
            st.success("Cache refreshed!")
            st.rerun()

# Main Application
def main():
    try:
        load_css()
        init_session_state()
        
        with st.sidebar:
            st.markdown("### 🧭 Navigation")
            pages = ["📊 Dashboard", "🧠 AI Assistant", "⚙️ Settings"]
            for page in pages:
                page_name = page.split(" ", 1)[1]
                if st.button(page, key=page_name, use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()
        
        # Page Routing
        if st.session_state.current_page == "Dashboard":
            render_dashboard()
        elif st.session_state.current_page == "AI Assistant":
            render_chat()
        elif st.session_state.current_page == "Settings":
            render_settings()
        
        st.markdown("<hr><div style='text-align:center;color:#666;'><p>🧠 Enhanced SupaBot with Smart Visualizations | Powered by Claude Sonnet 3.5</p></div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check your configuration and try refreshing the page.")

if __name__ == "__main__":
    main()