import streamlit as st
import pandas as pd
import altair as alt
import streamlit_option_menu
from streamlit_option_menu import option_menu
# from numerize import numerize
import plotly.express as px
import pandasql as ps

st.set_page_config(layout='wide')

df = pd.read_csv('streamlit_app\data\superstore.csv')

df['order_date'] = pd.to_datetime(df['order_date'])
df['ship_date'] = pd.to_datetime(df['ship_date'])
df['order_year'] = df['order_date'].dt.year

with st.sidebar:
    selected_year = st.selectbox("Select Year", sorted(df['order_year'].unique(), reverse=True), index=0)
    selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home","Geography Insight","Customer Segmentation","Product Analysis","Query Optimization and Processing","Contact Us"],
    icons = ["house","geo-alt","person-circle",'bi-archive',"snowflake","envelope"],
    menu_icon = "cast",
    default_index = 0
    
    )


CURR_YEAR = selected_year
PREV_YEAR = CURR_YEAR - 1

if selected == "Home":
    st.title("Tokopaedi Dashboard")

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

    mx_sales, mx_order, mx_customer, mx_profit_pct = st.columns(4)

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

        st.metric("Customer_id", value=(curr_cust), delta=f'{order_diff_pct:.2f}%')

    with mx_profit_pct:
        curr_profit_pct = data.loc[data['order_year']==CURR_YEAR, 'profit_pct'].values[0]
        prev_profit_pct = data.loc[data['order_year']==PREV_YEAR, 'profit_pct'].values[0]
        
        profit_pct_diff_pct = 100.0 * (curr_profit_pct - prev_profit_pct) / prev_profit_pct
        st.metric("profit_pct", value=f'{curr_profit_pct:.2f}%', delta=f'{profit_pct_diff_pct:.2f}%')


    freq = st.selectbox("Freq", ['Harian','Mingguan','Quartal','Bulanan'])

    timeUnit = {
        'Harian':'yearmonthdate',
        'Mingguan':'yearweek',
        'Quartal':'yearquarter',
        'Bulanan':'yearmonth'
    }

    st.header("Sales trend")
    # altair membuat object berupa chart dengan data di dalam parameter
    sales_line = alt.Chart(df[df['order_year']==CURR_YEAR]).mark_line().encode(
        alt.X('order_date', title='Order Date', timeUnit=timeUnit[freq]),
        alt.Y('sales', title='Revenue', aggregate='sum')
    )

    st.altair_chart(sales_line, use_container_width=True)

    st.header("Profit Distribution by Category and Segment")
    profit_bar = alt.Chart(df[df['order_year'] == CURR_YEAR]).mark_bar().encode(
    alt.X('order_date', title='Order Date', timeUnit='month'),
    alt.Y('profit:Q', title='Profit', aggregate='sum'),
    color=alt.Color('segment:N', title='segment')  # add color parameter here
    ).facet(
        column=alt.Column('ship_mode:N', title='Ship Mode')  # add facet parameter here
    )

    st.altair_chart(profit_bar, use_container_width=True)

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

if selected == "Geography Insight":

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
    freq = st.selectbox("Freq", ['Harian','Mingguan','Quartal','Bulanan'])

    timeUnit = {
         'Harian':'yearmonthdate',
        'Mingguan':'yearweek',
        'Quartal':'yearquarter',
        'Bulanan':'yearmonth'
    }

    col1 , col2 = st.columns((2)) 
    with col1:
        filtered_df = df[(df['region'] == 'East') & (df['order_year'] == CURR_YEAR)]
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
    
        profit_line = alt.Chart(filtered_df[filtered_df['order_year']==CURR_YEAR]).mark_line().encode(
        alt.X('order_date', title='order date', timeUnit=timeUnit[freq]),
        alt.Y('sales', title='Revenue', aggregate='sum')
        ).properties(
            width=500,  # Sesuaikan dengan lebar yang diinginkan
            height=600,
            title=f'Time Series Region: {region_name} ({CURR_YEAR})'
        ).configure_title(
            anchor='middle'
        )
        st.altair_chart(profit_line, use_container_width=True)

    col1 , col2 = st.columns((2)) 

    with col1:
        filtered_df = df[(df['region'] == 'South') & (df['order_year'] == CURR_YEAR)]
        region_name = filtered_df['region'].iloc[0]
        top_city = filtered_df.groupby('city')['sales'].sum().reset_index()
        top_city['sales'] = top_city['sales'].round(1)
        top_city.sort_values(['sales'], ascending=False, inplace=True)
        top_10_city = top_city.head(10)

        fig_2 = px.bar(top_10_city, y='sales', x='city', title=f'Akumulasi Sales Region :{region_name} ({CURR_YEAR})',
                            labels={'sales': 'Sales Amount', 'city': 'City'},
                            text_auto=".2f",
                            color='city')  # Replace 'region' with the actual column you want to use for color
        fig_2.update_layout(width=600, height=600, title_x=0.2, title_y=1)
        st.plotly_chart(fig_2, use_container_width=True)
    with col2:
        profit_line = alt.Chart(filtered_df[filtered_df['order_year']==CURR_YEAR]).mark_line().encode(
        alt.X('order_date', title='order date', timeUnit=timeUnit[freq]),
        alt.Y('sales', title='Revenue', aggregate='sum')
        ).properties(
            width=500,  # Sesuaikan dengan lebar yang diinginkan
            height=600,
            title=f'Time Series Region: {region_name} ({CURR_YEAR})'
        ).configure_title(
            anchor='middle'
        )
        st.altair_chart(profit_line, use_container_width=True)

    col1 , col2 = st.columns((2)) 
    with col1:

        filtered_df = df[(df['region'] == 'West') & (df['order_year'] == CURR_YEAR)]
        region_name = filtered_df['region'].iloc[0]
        top_city = filtered_df.groupby('city')['sales'].sum().reset_index()
        top_city['sales'] = top_city['sales'].round(1)
        top_city.sort_values(['sales'], ascending=False, inplace=True)
        top_10_city = top_city.head(10)

        fig_3 = px.bar(top_10_city, y='sales', x='city', title=f'Akumulasi Sales Region :{region_name} ({CURR_YEAR})',
                            labels={'sales': 'Sales Amount', 'city': 'City'},
                            text_auto=".2f",
                            color='city')  # Replace 'region' with the actual column you want to use for color
        fig_3.update_layout(width=800, height=650, title_x=0.2, title_y=1)
        st.plotly_chart(fig_3, use_container_width=True)

    with col2:
        profit_line = alt.Chart(filtered_df[filtered_df['order_year']==CURR_YEAR]).mark_line().encode(
        alt.X('order_date', title='order date', timeUnit=timeUnit[freq]),
        alt.Y('sales', title='Revenue', aggregate='sum')
        ).properties(
            width=500,  # Sesuaikan dengan lebar yang diinginkan
            height=600,
            title=f'Time Series Region: {region_name} ({CURR_YEAR})'
        ).configure_title(
            anchor='middle'
        )
        st.altair_chart(profit_line, use_container_width=True)


    col1 , col2 = st.columns((2)) 
    with col1:
        filtered_df = df[(df['region'] == 'Central') & (df['order_year'] == CURR_YEAR)]
        region_name = filtered_df['region'].iloc[0]
        top_city = filtered_df.groupby('city')['sales'].sum().reset_index()
        top_city['sales'] = top_city['sales'].round(1)
        top_city.sort_values(['sales'], ascending=False, inplace=True)
        top_10_city = top_city.head(10)

        fig_4 = px.bar(top_10_city, y='sales', x='city', title=f'Akumulasi Sales Region :{region_name} ({CURR_YEAR})',
                            labels={'sales': 'Sales Amount', 'city': 'City'},
                            text_auto=".2f",
                            color='city')  # Replace 'region' with the actual column you want to use for color
        fig_4.update_layout(width=800, height=650, title_x=0.2, title_y=1)
        st.plotly_chart(fig_4, use_container_width=True)

    with col2:
        profit_line = alt.Chart(filtered_df[filtered_df['order_year']==CURR_YEAR]).mark_line().encode(
        alt.X('order_date', title='order date', timeUnit=timeUnit[freq]),
        alt.Y('sales', title='Revenue', aggregate='sum')
        ).properties(
            width=500,  # Sesuaikan dengan lebar yang diinginkan
            height=600,
            title=f'Time Series Region: {region_name} ({CURR_YEAR})'
        ).configure_title(
            anchor='middle'
        )
        st.altair_chart(profit_line, use_container_width=True)

if selected == "Customer Segmentation":

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


    filtered_df = df[(df['order_year'] == CURR_YEAR)]
    # Calculating recency
    recency_df = filtered_df.groupby('customer_name', as_index=False)['order_date'].max()
    recent_date = recency_df['order_date'].max()
    recency_df['Recency'] = recency_df['order_date'].apply(
    lambda x: (recent_date - x).days)
    recency_df.rename(columns={'order_date':'Last Purchase Date'}, inplace=True)

    # Calculating Frequency
    frequency_df = filtered_df.groupby('customer_name', as_index=False)['order_date'].count()
    frequency_df.rename(columns={'order_date':'Frequency'}, inplace=True)

    # Calculating monetary
    monetary_df = filtered_df.groupby('customer_name', as_index=False)['sales'].sum()
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

    # Update chart to display values
    fig_cust_rfm.update_traces(textinfo='percent+label', text=rank_df['Customer Segment'].value_counts().values)

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

if selected == "Product Analysis":
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