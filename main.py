import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib as plt
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
# from numerize import numerize
import plotly.express as px
import pandasql as ps
import json
import statsmodels.api as sm
import warnings
from plotly.subplots import make_subplots
from st_functions import st_button, load_css
from PIL import Image

warnings.filterwarnings('ignore')

st.set_page_config(layout='wide')

df = pd.read_csv('streamlit_app\data\superstore.csv')

with open("gz_2010_us_040_00_20m.json") as response:
    geo = json.load(response)


df['order_date'] = pd.to_datetime(df['order_date'], format='%Y-%m-%d')
df['ship_date'] = pd.to_datetime(df['ship_date'])
df['order_year'] = df['order_date'].dt.year
df['days to ship'] = abs(df['ship_date']- df['order_date']).dt.days


with st.sidebar:
    selected_year = st.selectbox("Select Year", sorted(df['order_year'].unique(), reverse=True), index=0)
    selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home","Geography Insight","Customer Segmentation","Product Analysis","Predictive Analysis","Contact Us"],
    icons = ["house","geo-alt","person-circle",'bi-archive',"bi-bar-chart-fill","envelope"],
    menu_icon = "cast",
    default_index = 0
    
    )

CURR_YEAR = selected_year
PREV_YEAR = CURR_YEAR - 1

if selected == "Home" and selected_year < 2015:
    st.warning("Halaman Home hanya dapat memfilter minimal tahun 2015")
else:
    st.title("Tokopaedi Dashboard")
if selected == "Home":
    # 1 periksa tahun terakhir dari data
    # itung total sales, banyaknya order, banyaknya kosumen, profit %
    # di tahun tersebut

    data = pd.pivot_table(
        data=df,
        index='order_year',
        aggfunc={
            'sales':'sum',
            'profit':'sum',
            'order_id':pd.Series.nunique,
            'customer_id':pd.Series.nunique
        }
    ).reset_index()

    data['profit_pct'] = 100.0 * data['profit'] / data['sales']

    mx_sales, mx_order, mx_customer, mx_profit_pct, gauge_shipping = st.columns(5)

    def format_big_number(num):
        if num >= 1e6:
            return f"{num / 1e6:.2f} Mio"
        elif num >= 1e3:
            return f"{num / 1e3:.2f} K"
        else:
            return f"{num:.2f}"

    with mx_sales:

        curr_sales = data.loc[data['order_year']==CURR_YEAR, 'sales'].values[0]
        prev_sales = data.loc[data['order_year']==PREV_YEAR, 'sales'].values[0]
        
        sales_diff_pct = 100.0 * (curr_sales - prev_sales) / prev_sales

        st.metric("Sales", value=format_big_number(curr_sales), delta=f'{sales_diff_pct:.2f}%')

    with mx_order:
        curr_order = data.loc[data['order_year'] == CURR_YEAR, 'order_id'].values[0]
        prev_order = data.loc[data['order_year'] == PREV_YEAR, 'order_id'].values[0]

        order_diff_pct = 100.0 * (curr_order - prev_order) / prev_order

        st.metric("Order", value=format_big_number(curr_order), delta=f'{order_diff_pct:.2f}%')

    with mx_customer:
        curr_cust = data.loc[data['order_year'] == CURR_YEAR, 'customer_id'].values[0]
        prev_cust = data.loc[data['order_year'] == PREV_YEAR, 'customer_id'].values[0]

        order_diff_pct = 100.0 * (curr_cust - prev_cust) / prev_cust

        st.metric("Jumlah Customer", value=(curr_cust), delta=f'{order_diff_pct:.2f}%')

    with mx_profit_pct:
        curr_profit_pct = data.loc[data['order_year']==CURR_YEAR, 'profit_pct'].values[0]
        prev_profit_pct = data.loc[data['order_year']==PREV_YEAR, 'profit_pct'].values[0]
        
        profit_pct_diff_pct = 100.0 * (curr_profit_pct - prev_profit_pct) / prev_profit_pct
        st.metric("profit_pct", value=f'{curr_profit_pct:.2f}%', delta=f'{profit_pct_diff_pct:.2f}%')


    with gauge_shipping:
        filtered_df = df[df['order_year'] == selected_year]
        value =int(np.round(filtered_df['days to ship'].mean()))  # Example value

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': "Average Shipping Days", 'font': {'size': 16}},
            gauge={'axis': {'range': [filtered_df['days to ship'].min() , filtered_df['days to ship'].max()]},
                'bar': {'color': "#005C53"},
                }
        ))

        fig.update_layout(width=150, height=150, margin=dict(l=10, r=10, b=10, t=4))

        st.plotly_chart(fig)

    freq = st.selectbox("Freq", ['Harian','Mingguan','Bulanan','Quartal'])

    timeUnit = {
        'Harian':'yearmonthdate',
        'Mingguan':'yearweek',
        'Quartal':'yearquarter',
        'Bulanan':'yearmonth'
    }

    st.header("Sales trend")
    # altair membuat object berupa chart dengan data di dalam parameter
    # Reshape the data
    df_filter = df.loc[:, ['order_date','order_year', 'sales', 'profit']].reset_index(drop=True)
    df_merge = pd.melt(df_filter, id_vars=['order_date','order_year'], value_vars=['sales', 'profit'], var_name='metric', value_name='value')
    df_sales = df_merge.loc[df_merge['metric'] == 'sales']
    df_profit = df_merge.loc[df_merge['metric'] == 'profit']
    
    colors = ['#5276A7','#F18727']
    base = alt.Chart(df_sales[df_sales['order_year'] == CURR_YEAR]).encode(
    x=alt.X('order_date:T', title='Order Date', timeUnit=timeUnit[freq]),
    color=alt.Color('metric', scale=alt.Scale(domain=['sales', 'profit'], range=['#5276A7', '#F18727']), legend=alt.Legend(
        orient='none',
        legendX=380, legendY=-5,
        direction='horizontal',
        titleAnchor='middle'))
    )
    bar_chart_A = base.mark_bar(color='#5276A7').encode(
        alt.Y('sum(value):Q', axis=alt.Axis(title='Sales', titleColor='#5276A7')),
        tooltip=[alt.Tooltip('sum(value)')]
    )

    base_1 = alt.Chart(df_profit[df_profit['order_year'] == CURR_YEAR]).encode(
    x=alt.X('order_date:T', title='Order Date', timeUnit=timeUnit[freq]),
    
    )
    line_chart_B = base_1.mark_line(stroke='#F18727', interpolate='monotone').encode(
        alt.Y('sum(value):Q', axis=alt.Axis(title='Profit', titleColor='#F18727')),
        tooltip=[alt.Tooltip('sum(value)', title='Profit')],
    )

    dual_axis = alt.layer(bar_chart_A, line_chart_B).resolve_scale(y='independent')

    # Assuming st is your Streamlit module
    st.altair_chart(dual_axis, use_container_width=True)

    # Bikin 4 kolom berisi sales dari tiap kategori
    # Setiap kolom mewakili region yang berbeda

    # st.subheader("Sales Distribution by Category and Segment")
    # bar_chart = alt.Chart(df[df['order_year']==CURR_YEAR]).mark_bar().encode(
    #     column='category:N',
    #     y='sum(sales):Q',
    #     color='segment:N',
    #     x='segment:N'
    # ).properties(width=350,height=220)
    # st.altair_chart(bar_chart)
    
    # Jika ingin menggunakan lebih presisi tengah/kanan/kiri
    # __, midcol, leftcol = st.columns([1,2,1])

    # with midcol:
    #     st.header("Sales vs Profit Correlation")
    #     scatter = alt.Chart(df[df['order_year']==CURR_YEAR]).mark_point().encode(
    #         y='sales:Q',
    #         x='profit:Q',   
    #         color='region:N',
    #     )
    #     st.altair_chart(scatter, use_container_width=True)

    # with leftcol:
    #     st.subheader("Ship Mode wise sales")
    #     fig = px.pie(df, values="sales", names="region", hole=0.5)
    #     fig.update_traces(text=df["region"], textposition="outside") 
    #     st.plotly_chart(fig, use_container_width=True) 

    col1 , col2 = st.columns((2)) 

    with col1:
        st.subheader("Sales vs Profit Correlation")
        scatter = alt.Chart(df[df['order_year']==CURR_YEAR]).mark_point().encode(
            y='profit:Q',
            x='sales:Q',  
            color='region:N',
        )
        st.altair_chart(scatter, use_container_width=True)

    with col2:
        st.subheader("Ship Mode wise sales")
        fig = px.pie(df, values="sales", names="ship_mode", hole=0.5)
        fig.update_traces(text=df["ship_mode"], textposition="outside") 
        st.plotly_chart(fig, use_container_width=True) 


    st.subheader("Hierarchical view od Sales using TreeMap")
    filtered_df = df[df['order_year'] == selected_year]  # filter dataframe based on selected year
    fig3 = px.treemap(filtered_df, path=["order_year","region", "category", "subcategory"], values= "sales",
                    color= "subcategory")
    fig3.update_layout(width= 800, height= 650)
    st.plotly_chart(fig3,use_container_width=True )

elif selected == "Geography Insight":

    col1 , col2 = st.columns((2)) 

    with col1:
        st.header("Region Sales")
        filtered_df = df[df['order_year'] == selected_year] 
        fig = px.pie(filtered_df, values="sales", names="region", hole=0.5)
        fig.update_traces(text=filtered_df["region"], textposition="outside") 
        st.plotly_chart(fig, use_container_width=True) 

    with col2:
        st.subheader("Diagnostic Geographic Sales")
        filtered_df = df[df['order_year'] == selected_year] 
        top_city = filtered_df.groupby(['city','region'])['sales'].sum().reset_index()
        top_city['sales'] = top_city['sales'].round(1)
        top_city.sort_values(['sales'], ascending=False, inplace=True)
        top_10_city = top_city.head(10)

        fig = px.bar(top_10_city, y='sales', x='city', title='Top 10 City by Sales',
                    labels={'sales': 'Sales Amount', 'city': 'City'},
                    text_auto=".2s",
                    color='region')  # Replace 'region' with the actual column you want to use for color
        fig.update_layout(width=800, height=500)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales Distribution by City")
    
    col1 , col2 = st.columns((2)) 
    with col1:
        freq = st.selectbox("Freq", ['Harian','Mingguan','Bulanan','Quartal'])

        timeUnit = {
            'Harian':'yearmonthdate',
            'Mingguan':'binnedyearweek',
            'Quartal':'yearquarter',
            'Bulanan':'yearmonth'
        }
    with col2:
        selected_region = st.selectbox("Select Region", df['region'].unique())
    col1 , col2 = st.columns((2)) 
    with col1:
        filtered_df = df[(df['region'] == selected_region) & (df['order_year'] == CURR_YEAR)]
        region_name = filtered_df['region'].iloc[0]
        top_city = filtered_df.groupby('city')['sales'].sum().reset_index()
        top_city['sales'] = top_city['sales'].round(1)
        top_city.sort_values(['sales'], ascending=False, inplace=True)
        top_10_city = top_city.head(10)

        fig_1 = px.bar(top_10_city, y='sales', x='city', title=f'Akumulasi Sales Region :{region_name} ({CURR_YEAR})',
                            labels={'sales': 'Sales Amount', 'city': 'City'},
                            text_auto=".2f",
                            color='city')  
        fig_1.update_layout(width=600, height=600,title_x=0.2, title_y=1)
        st.plotly_chart(fig_1, use_container_width=True)


    with col2:
        geometries = pd.read_csv('georef-united-states-of-america-zc-point.csv', sep = ";")
        merged_df = pd.merge(df, geometries, left_on='postal_code', right_on='Zip Code', how='left')
        merged_df[['latitude', 'longitude']] = merged_df['Geo Point'].str.split(',', expand=True)
        merged_df['latitude'] = merged_df['latitude'].astype(float)
        merged_df['longitude'] = merged_df['longitude'].astype(float)
        merged_df = merged_df[['order_year','region','state','latitude','longitude','profit']]
        
        filtered_df = merged_df[(merged_df['region'] == selected_region) & (df['order_year'] == CURR_YEAR)]
        center_lat = filtered_df['latitude'].mean()
        center_lon = filtered_df['longitude'].mean()
        filtered_df = filtered_df.groupby(['order_year',"state",'latitude','longitude']).agg(
            {"profit": "sum"}
        )
        filtered_df.reset_index(inplace=True)

        fig = go.Figure(
            go.Choroplethmapbox(
                geojson=geo,
                locations=filtered_df.state,
                featureidkey="properties.NAME",
                z=filtered_df.profit,
                colorscale="sunsetdark",
                # zmin=0,
                # zmax=500000,
                marker_opacity=0.5,
                marker_line_width=0,
            )
        )
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=3.5,
            mapbox_center={"lat": center_lat , "lon": center_lon},
            width=600,
            height=600,
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig)
        
        
    filtered_df = df[(df['region'] == selected_region) & (df['order_year'] == CURR_YEAR)]
    df_filter = filtered_df.loc[:, ['order_date','order_year', 'sales', 'profit']].reset_index(drop=True)
    df_merge = pd.melt(df_filter, id_vars=['order_date','order_year'], value_vars=['sales', 'profit'], var_name='metric', value_name='value')
    df_sales = df_merge.loc[df_merge['metric'] == 'sales']
    df_profit = df_merge.loc[df_merge['metric'] == 'profit']
        
    colors = ['#5276A7','#F18727']
    base = alt.Chart(df_sales[df_sales['order_year'] == CURR_YEAR]).encode(
    x=alt.X('order_date:T', title='Order Date', timeUnit=timeUnit[freq]),
    tooltip=['order_date','sum(value)'],
    color=alt.Color('metric', scale=alt.Scale(domain=['sales', 'profit'], range=['#5276A7', '#F18727']), legend=alt.Legend(
            orient='none',
            legendX=100, legendY=-5,
            direction='horizontal',
            titleAnchor='middle'))
        )
    bar_chart_A = base.mark_bar(color='#5276A7').encode(
            alt.Y('sum(value):Q', axis=alt.Axis(title='Sales', titleColor='#5276A7')),
        )

    base_1 = alt.Chart(df_profit[df_profit['order_year'] == CURR_YEAR]).encode(
        x=alt.X('order_date:T', title='Order Date', timeUnit=timeUnit[freq]),
         
        )
    line_chart_B = base_1.mark_line(stroke='#F18727', interpolate='monotone').encode(
            alt.Y('sum(value):Q', axis=alt.Axis(title='Profit', titleColor='#F18727')),
          
        )
    
    dual_axis = alt.layer(bar_chart_A,line_chart_B).resolve_scale(y='independent')
        
    chart_title= f"Sales and Profit Over Time {region_name} ({CURR_YEAR})"
        # Assuming st is your Streamlit module
    title_params = alt.TitleParams(
    text=chart_title,
    align='center',  # Center-align the title
    anchor='middle'  # Anchor the title in the middle
    )

    # Display the chart with title
    chart_with_title = dual_axis.properties(width=600, height=600, title=title_params)
    st.altair_chart(chart_with_title, use_container_width=True)
  
elif selected == "Customer Segmentation":

    filtered_df = df[(df['order_year'] == CURR_YEAR)]
    best_customer = pd.pivot_table(
    data=filtered_df,
    index='customer_name',
    aggfunc={
        'sales': 'sum'
    } ).reset_index()
    best_customer = best_customer.sort_values(by=['sales'], ascending=False).head(10)
    best_customer['sales'] = best_customer['sales'].round(2)

    group_cat_customer = pd.pivot_table(
    data=filtered_df,
    index=['customer_name', 'category'],
    aggfunc={
        'sales': 'sum'
        }
    ).reset_index()
    merged_df = pd.merge(best_customer, group_cat_customer, on='customer_name', how='left')

    fig_best_cust = px.bar(merged_df, y='customer_name', x='sales_x', title=f'Top 10 Best Customer:({CURR_YEAR})',
                            labels={'sales_x': 'Sales Amount', 'customer_name': 'Customer Name'},
                            text_auto=".2f",
                            color='category')  # Replace 'category' with the actual column you want to use for color
    fig_best_cust.update_layout(width=600, height=600, title_x = 0.5)
    fig_best_cust.update_yaxes(categoryorder='total ascending')
    st.plotly_chart(fig_best_cust, use_container_width=True)


    # filtered_df = df[(df['order_year'] == CURR_YEAR)]
    # Calculating recency
    recency_df = df.groupby('customer_name', as_index=False)['order_date'].max()
    recent_date = recency_df['order_date'].max()
    recency_df['Recency'] = recency_df['order_date'].apply(
    lambda x: (recent_date - x).days)
    recency_df.rename(columns={'order_date':'Last Purchase Date'}, inplace=True)

    # Calculating Frequency
    frequency_df = df.groupby('customer_name', as_index=False)['order_date'].count()
    frequency_df.rename(columns={'order_date':'Frequency'}, inplace=True)

    # Calculating monetary
    monetary_df = df.groupby('customer_name', as_index=False)['sales'].sum()
    monetary_df.rename(columns={'sales':'Monetary'}, inplace=True)

    # Merging all three df in one df
    rfm_df = recency_df.merge(frequency_df, on='customer_name')
    rfm_df = rfm_df.merge(monetary_df, on='customer_name')
    rfm_df['Monetary'] = rfm_df['Monetary'].round(2)
    rfm_df.drop(['Last Purchase Date'], axis=1, inplace=True)

    rank_df = rfm_df.copy() # We make copy of rfm_df because we will need RFM features later

    # Normalizing the rank of the customers
    rank_df['r_rank'] = rank_df['Recency'].rank(ascending=False)
    rank_df['f_rank'] = rank_df['Frequency'].rank(ascending=False)
    rank_df['m_rank'] = rank_df['Monetary'].rank(ascending=False)

    rank_df['r_rank_norm'] = (rank_df['r_rank'] / rank_df['r_rank'].max()) * 100
    rank_df['f_rank_norm'] = (rank_df['f_rank'] / rank_df['f_rank'].max()) * 100
    rank_df['m_rank_norm'] = (rank_df['m_rank'] / rank_df['m_rank'].max()) * 100

    rank_df.drop(['r_rank','f_rank','m_rank'], axis=1, inplace=True)

    # Calculating RFM scores
    rank_df['rfm_score'] = (0.15*rank_df['r_rank_norm']) + (0.28*rank_df['f_rank_norm']) + (0.57*rank_df['m_rank_norm'])
    rank_df = rank_df[['customer_name','rfm_score']]
    rank_df['rfm_score'] = round(rank_df['rfm_score']*0.05, 2)

    # Masking all customers rfm scores by rating conditions to set customer segments easily
    top_customer_mask = (rank_df['rfm_score'] >= 4.5)
    high_value_mask = ((rank_df['rfm_score']<4.5) & (rank_df['rfm_score']>=4))
    medium_value_mask = ((rank_df['rfm_score']<4) & (rank_df['rfm_score']>=3))
    low_value_mask = ((rank_df['rfm_score']<3) & (rank_df['rfm_score']>=1.6))
    lost_mask = (rank_df['rfm_score'] < 1.6)

    colors = ['#3C0753', '#F0F3FF','#910A67', '#720455', '#030637']

    rank_df.loc[top_customer_mask, 'Customer Segment'] = 'Top Customer'
    rank_df.loc[high_value_mask, 'Customer Segment'] = 'High Value Customer'
    rank_df.loc[medium_value_mask, 'Customer Segment'] = 'Medium Value Customer'
    rank_df.loc[low_value_mask, 'Customer Segment'] = 'Low Value Customer'
    rank_df.loc[lost_mask, 'Customer Segment'] = 'Lost Customer'

    group_rfm_customer = pd.pivot_table(
        data=rank_df,
        index='Customer Segment',
        values='rfm_score',
        aggfunc='mean'
    ).reset_index()
    group_rfm_customer['rfm_score'] = group_rfm_customer['rfm_score'].round(2)

    st.subheader("RFM Segmentation")
    fig_cust_rfm = px.pie(group_rfm_customer, values="rfm_score",
                        names="Customer Segment",
                        title='Customer Segments')

    # Update chart to display values and set colors
    fig_cust_rfm.update_traces(
        textinfo='percent+label',
        text=rank_df['Customer Segment'].value_counts().values,
        marker=dict(colors=colors)
    )

# Optional: Add a title to the layout
    fig_cust_rfm.update_layout(title='Customer Segment Distribution')

    # Display the plot using st.plotly_chart()
    st.plotly_chart(fig_cust_rfm, use_container_width=True)

    st.subheader("Exploration Lost Customer Detail")
    merged_rfm_cust = pd.merge(rank_df, group_rfm_customer, on='Customer Segment', how='left')
    lost_customer_df = merged_rfm_cust.loc[merged_rfm_cust['Customer Segment'] == 'Lost Customer']
    lost_customer_df_detail = pd.merge(lost_customer_df, df, on='customer_name', how='left')
    lost_customer_df_detail = lost_customer_df_detail.loc[:, ['customer_name', 'country', 'city', 'state', 'postal_code', 'region', 'category', 'subcategory','rfm_score_x']].reset_index(drop=True)
    
    # Area input untuk menulis query SQL
    query_input = st.text_area('Masukkan Query SQL', 'SELECT * FROM lost_customer_df_detail LIMIT 5')

    # Tombol eksekusi query
    if st.button('Eksekusi Query'):
        try:
            # Menjalankan query dan menampilkan hasilnya
            result_df = ps.sqldf(query_input, locals())
            st.write('Hasil Query:')
            st.write(result_df)
        except Exception as e:
            st.error(f'Error: {e}')

elif selected == "Product Analysis":
    st.header("Product Anaysis")

    freq = st.selectbox("Freq", ['Harian','Mingguan','Quartal','Bulanan'])

    timeUnit = {
         'Harian':'yearmonthdate',
        'Mingguan':'yearweek',
        'Quartal':'yearquarter',
        'Bulanan':'yearmonth'
    }
    categories = ['Accessories', 'Appliances', 'Art', 'Binders', 'Bookcases',
                'Chairs', 'Copiers', 'Envelopes', 'Fasteners', 'Furnishings',
                'Labels', 'Machines', 'Paper', 'Phones', 'Storage', 'Supplies',
                'Tables']

    fixed_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#000000', '#00ff7f', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    color_discrete_map = {category: color for category, color in zip(categories, fixed_colors)}
    color_scale = alt.Scale(domain=categories, range=fixed_colors)
    
    st.subheader("Profit Distribution by Sub Category")
    profit_bar = alt.Chart(df[df['order_year'] == CURR_YEAR]).mark_bar().encode(
    alt.X('order_date', title='Order Date', timeUnit=timeUnit[freq]),
    alt.Y('profit:Q', title='Profit', aggregate='sum'),
    color=alt.Color('subcategory:N', title='Sub-Categories',scale=color_scale )  # add color parameter here
    ).facet(
        column=alt.Column('category', title='Category')  # add facet parameter here
    ).resolve_scale(
    y='independent'
)

    st.altair_chart(profit_bar, use_container_width=True)


    filtered_df = df[(df['order_year'] == CURR_YEAR)]
    top_prod = pd.pivot_table(
    data=filtered_df,
    index='subcategory',
    aggfunc={
        'sales': 'sum',
        'product_id': pd.Series.nunique
    } ).reset_index()
    top_prod = top_prod.rename(columns={'product_id': 'total'})
    top_10_prod = top_prod.sort_values(by=['sales'], ascending=False).head(10)
    top_10_prod['sales'] = top_10_prod['sales'].round(2)

    fig_top10prod = px.bar(top_10_prod, y='sales', x='subcategory', title=f'Top 10 Selling Sub Category: ({CURR_YEAR})',
                                labels={'sales': 'Sales Amount', 'subcategory': 'Sub-Categories'}, 
                                hover_data=['total'],
                                text_auto=".2f",
                                color='subcategory',
                                color_discrete_map=color_discrete_map )  # Replace 'region' with the actual column you want to use for color
    fig_top10prod.update_layout(width=800, height=650, title_x=0.5)
    st.plotly_chart(fig_top10prod, use_container_width=True)

    # pivot_table = pd.pivot_table(
    # filtered_df,
    # index=['subcategory', 'product_name'],
    # aggfunc={'sales': 'sum', 'profit': 'sum', 'product_id': df['product_id'].value_counts()}
    # ).reset_index()


    product_df = filtered_df.groupby(['subcategory', 'product_name']).agg({'sales': 'sum'}).round(2).reset_index()

    # Calculate the 'Amount' column per product_name within each category
    product_df['Amount'] = df.groupby(['subcategory', 'product_name'])['product_name'].transform('count')

    # Calculate the 'Price' column
    product_df['Price'] = round(product_df['sales'] / product_df['Amount'], 2)

    # Pivot the DataFrame
    pivot_table = pd.pivot_table(product_df, values=['sales', 'Amount', 'Price'], index=['subcategory', 'product_name'])

    top_10_sales = (
    product_df.groupby('subcategory', group_keys=False)
    .apply(lambda x: x.nlargest(3, 'sales'))
    .reset_index(drop=True)
    )

    
    st.subheader("Top 3 Orders Sales Product per SubCategory Pivot Table")
    subcategory_colors = {
    'Bookcases': '#ff7f00',
    'Chairs': '#ffff33',
    'Labels': 'lightcoral',
    'Tables': '#ffd16a',
    'Storage': '#984ea3',
    'Furnishings': '#0e1117',
    'Art': 'lightgoldenrodyellow',
    'Phones': '#4daf4a',
    'Binders': '#984ea3',
    'Appliances': '#377eb8',
    'Paper': 'lightpink',
    'Accessories': '#e41a2c',
    'Envelopes': 'lightseagreen',
    'Fasteners': 'lightcyan',
    'Supplies': 'lightsteelblue',
    'Machines': '#e41a1c',
    'Copiers': '#a65628'
    }

    def row_style(row, subcategory_colors):
        subcategory = row['subcategory']
        color = subcategory_colors.get(subcategory, '')

        # Style headers with black color
        if isinstance(row.name, tuple) and row.name[0] == '':  # Header cell
            return [
                f'background-color: {color}; font-weight: bold; color: black'
            ] * len(row)
        else:  # Value cell
            if subcategory == 'Furnishings':  # Special styling for 'Furnishings'
                return [
                    f'background-color: {color}; color: white'  # Yellow background for 'Furnishings'
                ] * len(row)
            else:
                return [
                    f'background-color: {color}; color: black'
                ] * len(row)

    # Assuming you have your `top_10_sales` DataFrame ready

    styled_pivot_table = top_10_sales.style.apply(
        lambda row: row_style(row, subcategory_colors), axis=1
    )
    container = st.container()

    # Atur margin container untuk memusatkan tabel
    with container:
        st.dataframe(styled_pivot_table, width=1000, height=1000)

elif selected == "Predictive Analysis":
        st.header("Predict Time Series Category Sales")
        selected_cat = st.selectbox("Select Category", df['category'].unique())

        filtered_df = df[(df['category'] == selected_cat)]
        filtered_df = filtered_df[['order_date','sales']]

        filtered_df['order_date'] = pd.to_datetime(filtered_df['order_date'])
        filtered_df = filtered_df.sort_values('order_date')
        filtered_df = filtered_df.set_index('order_date')

        value_y = filtered_df['sales'].resample('MS').mean()

        decomposition_value= sm.tsa.seasonal_decompose(value_y,model='additive')
        # Create plotly figure
        fig_names = ['Observed', 'Trend', 'Seasonal', 'Residual']
        decomposition_values = [decomposition_value.observed, decomposition_value.trend, decomposition_value.seasonal, decomposition_value.resid]
        st.subheader("Decomposition Plot")
    
        fig = make_subplots(rows=2, cols=2, subplot_titles=fig_names)

        # Loop through each figure type
        for i, (fig_name, values) in enumerate(zip(fig_names, decomposition_values), start=1):
            # Calculate row and col indices
            row_idx, col_idx = divmod(i - 1, 2)  # Subtract 1 to start index from 0

            # Add trace to subplot
            fig.add_trace(go.Scatter(x=values.index, y=values, mode='lines' if fig_name != 'Residual' else 'markers',
                                    marker=dict(size=5, color='red') if fig_name == 'Residual' else None, name=fig_name),
                        row=row_idx + 1, col=col_idx + 1)

            # Update layout
            fig.update_xaxes(title_text='Date', row=row_idx + 1, col=col_idx + 1)
            fig.update_yaxes(title_text='Sales', row=row_idx + 1, col=col_idx + 1, type='log' if fig_name == 'Residual' else None)

        # Update overall layout
        fig.update_layout(height=600, width=800, showlegend=False)

        # Display plotly figure
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("SARIMA Parameter")
        best_params = {'Furniture': {'p': 0, 'd': 1, 'q': 1, 'P': 0, 'D': 1, 'Q': 1, 's': 12, 'AIC': 251.24707755083713},
               'Office Supplies': {'p': 0, 'd': 1, 'q': 1, 'P': 0, 'D': 1, 'Q': 1, 's': 12, 'AIC': 231.55283799226925},
               'Technology': {'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 1, 'Q': 1, 's': 12, 'AIC': 294.1604201223718}}

        # Streamlit selectbox to choose category

        # Display the best parameters for the selected category
        if selected_cat in best_params:
            best_param_values = best_params[selected_cat]
            st.write(f"Best parameters for {selected_cat}: {best_param_values}")

            # Extracting parameters
            p, d, q, P, D, Q, s = best_param_values['p'], best_param_values['d'], best_param_values['q'], \
                                best_param_values['P'], best_param_values['D'], best_param_values['Q'], best_param_values['s']

            # Assuming value_y is defined earlier
            mod_value = sm.tsa.statespace.SARIMAX(value_y,
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, s),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)

            results_value = mod_value.fit()

            # Now you can use 'results_value' for further analysis or predictions
            st.write("SARIMAX Model Results:")
            st.write(results_value.summary())
        else:
            st.write("Invalid category selected.")

        results_value = mod_value.fit()
        pred_value = results_value.get_prediction(start = pd.to_datetime('2017-01-01'), dynamic = False)
        pred_ci_value = pred_value.conf_int()
        value_forecasted = pred_value.predicted_mean
        value_truth = value_y['2017-01-01':]
        mse = ((value_forecasted - value_truth) ** 2).mean()
        rmse = mse**0.5
        st.subheader("Model Evaluations")
        st.write('MSE of forecast :{}'.format(round(mse,2)))
        st.write('RMSE of forecast :{}'.format(round(rmse,2)))


        st.subheader(f"Predict {selected_cat} Sales")
        forecast_period = st.selectbox("Select forecast period", ['1 year', '2 years', '3 years'])

        # Convert selected period to steps
        if forecast_period == '1 year':
            steps = 12
        elif forecast_period == '2 years':
            steps = 24
        elif forecast_period == '3 years':
            steps = 36
        else:
            st.error("Invalid forecast period selected. Please choose 1 year, 2 years, or 3 years.")

        # Get forecast values and confidence interval
        pred_uc_value = results_value.get_forecast(steps=steps)
        pred_ci_value = pred_uc_value.conf_int()

        total_sales_forecasted = pred_uc_value.predicted_mean.sum()

# Menampilkan total penjualan dalam periode yang dipilih
        st.write(f"Total sales forecasted for the selected period: {round(total_sales_forecasted, 2)}")

        # Plotting the forecast
        fig, ax = plt.subplots(figsize=(10, 8))
        value_y.plot(ax=ax, label='observed')
        pred_uc_value.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci_value.index,
                        pred_ci_value.iloc[:, 0],
                        pred_ci_value.iloc[:, 1], color='k', alpha=0.6)
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{selected_cat} Sales')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(fig, use_container_width=True)
            
elif selected == "Contact Us":
        load_css()
        col1, col2, col3 = st.columns(3)
        col2.image(Image.open('dp.png'))

        st.header('Muhammad Zaki Fuadi')

        st.info('Professional Data (Analyst, Engineer, Science)')

        icon_size = 20


        st_button('linkedin', 'https://www.linkedin.com/in/mzfuadi97/', 'Follow me on LinkedIn', icon_size)