"""
============================================================
FOOD ORDERING BEHAVIOR ANALYSIS
============================================================

Author: Garima Kamboj
Tools Used:
- Python
- Pandas
- Matplotlib
- Seaborn

Project Goals:
1. Analyze customer ordering behavior
2. Identify top-performing cities
3. Understand customer retention
4. Study cuisine preferences
5. Generate business recommendations

============================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# SECTION 1 : LOAD DATA
# =========================================================

df = pd.read_csv(
    r"D:\FoodOrderingBehaviorAnalysis\data\food_ordering_behavior_dataset.csv"
)

# =========================================================
# SECTION 2 : DATA OVERVIEW
# =========================================================

def dataset_overview(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nSummary Statistics:")
    print(df.describe())


# =========================================================
# SECTION 3: DATA CLEANING
# =========================================================

def clean_data(df):

    df = df.drop_duplicates()

    numeric_columns = [
        "user_id",
        "age",
        "order_value",
        "delivery_fee",
        "time_taken_to_order",
        "rating_given"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    text_columns = [
        "city",
        "cuisine",
        "meal_type",
        "restaurant_type",
        "mood",
        "company"
    ]

    for col in text_columns:

        if col in df.columns:

            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.title()
            )

    return df


# =========================================================
# SECTION 4 : HANDLE MISSING VALUES
# =========================================================

def handle_missing_values(df):

    numeric_columns = [
        "age",
        "order_value",
        "delivery_fee",
        "time_taken_to_order",
        "rating_given"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = df[col].fillna(
                df[col].median()
            )

    return df


# =========================================================
# SECTION 5 : FEATURE ENGINEERING
# =========================================================

def feature_engineering(df):

    df["Repeat Customer"] = (
        df.groupby("user_id")["user_id"]
        .transform("count") > 1
    )

    df["Customer Total Spend"] = (
        df.groupby("user_id")["order_value"]
        .transform("sum")
    )

    df["Customer Total Orders"] = (
        df.groupby("user_id")["order_id"]
        .transform("count")
    )

    df["Customer Segment"] = pd.cut(
        df["Customer Total Spend"],
        bins=[0,500,1000,2000,5000,10000],
        labels=[
            "Low",
            "Medium",
            "High",
            "Premium",
            "VIP"
        ]
    )

    return df



# =========================================================
# SECTION 6 : CITY ANALYSIS
# =========================================================

def city_analysis(df):
    """
    Analyze city-wise revenue,
    order count and average order value.
    """

    city_summary = (
        df.groupby("city")
        .agg(
            Revenue=("order_value","sum"),
            Orders=("order_id","count"),
            Avg_Order=("order_value","mean")
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
    )

    print(city_summary.head(5))
    plt.subplot(4,3,1)
    

    plt.bar(
    city_summary.index,
    city_summary["Revenue"].values,
    color="skyblue"
    )

    

    plt.title("Revenue by City")

    plt.xlabel("City")

    plt.ylabel("Revenue")

    plt.xticks(rotation=45)

    plt.grid(alpha=0.3)

    
    top_city = city_summary["Revenue"].idxmax()
    city_orders = (
    df.groupby("city")["order_id"]
      .count()
      .sort_values(ascending=False)
)

    print("\nOBSERVATION:")
    print(f"{top_city} generated the highest revenue.")

    print("\nBUSINESS INSIGHT:")
    print("Revenue is concentrated in a few cities.")

    print("\nRECOMMENDATION:")
    print("Increase marketing efforts in lower-performing cities.")

# =========================================================
# SECTION 7 : CUSTOMER ANALYSIS
# =========================================================

def customer_analysis(df):
    print("Unique Customers:",
      df["user_id"].nunique())

    top_customers = (
        df.groupby("user_id")["order_value"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print(top_customers.head())

    repeat_revenue = (
        df.groupby("Repeat Customer")
        ["order_value"]
        .sum()
    )

    print("\nRevenue by Customer Type")
    print(repeat_revenue.head())
    plt.subplot(4,3,2)
    plt.pie(
        repeat_revenue.values,
        labels=repeat_revenue.index,
        autopct="%1.1f%%"
    )

    plt.title(
        "Revenue Contribution by Customer Type"
    )

    
    print("""Observation

Repeat customers contribute a significant portion of total revenue.

Business Insight

Existing customers are more valuable than one-time customers.

Recommendation

Introduce loyalty programs, reward points, and personalized offers to increase retention.""")

# =========================================================
# SECTION 8 : CUISINE ANALYSIS
# =========================================================

def cuisine_analysis(df):

    cuisine_sales = (
        df.groupby("cuisine")
        ["order_value"]
        .sum()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print(cuisine_sales)
    plt.subplot(4,3,3)
    colors = plt.cm.viridis(
    range(len(cuisine_sales))
)

    

    plt.barh(
    cuisine_sales.index,
    cuisine_sales.values,
    color=colors
)

    for i, v in enumerate(cuisine_sales.values):

       plt.text(
        v,
        i,
        f"{v:,.0f}"
    )

    plt.title("Top 10 Cuisines")

    plt.xlabel("Revenue")

    plt.ylabel("Cuisine")

    
    cuisine_rating = (
    df.groupby("cuisine")["rating_given"]
      .mean()
      .sort_values(ascending=False)
)
    print("""Observation

Deserts cuisines generated the highest order value.

Business Insight

Customers show a strong preference for this cuisine, making them key revenue drivers.

Recommendation

Partner with more restaurants offering popular cuisines and provide cuisine-specific discounts.""")


# =========================================================
# SECTION 9 : DELIVERY ANALYSIS
# =========================================================

def delivery_analysis(df):

    delivery_summary = (
        df.groupby("delivery_fee")
        ["order_value"]
        .mean()
    )
    plt.subplot(4,3,4)
    sns.histplot(delivery_summary,bins=5,kde=True)
    plt.title("Delivery Fee Impact")
    

    print(delivery_summary.head())
    print("""Observation

Medium delivery fee categories generated the highest revenue.

Business Insight

Customers appear comfortable paying moderate delivery charges.

Recommendation

Avoid excessive delivery fees and optimize pricing to maximize order volume.""")
    
# =========================================================
# SECTION 10 : MEAL TYPE ANALYSIS
# =========================================================

def meal_type_analysis(df):

    print("\n" + "=" * 60)
    print("MEAL TYPE ANALYSIS")
    print("=" * 60)

    meal_sales = (
        df.groupby("meal_type")["order_value"]
        .sum()
        .sort_values(ascending=False)
    )

    print(meal_sales.head())
    plt.subplot(4,3,5)
    sns.barplot(x=meal_sales.index,y=meal_sales.values)
    plt.title("Meal Type Distribution")
    plt.xlabel("meals")
    plt.ylabel("Total Sales")
    
    top_meal = meal_sales.idxmax()

    print(f"""
Observation:
{top_meal} generates the highest sales.

Business Insight:
Customers strongly prefer this meal category.

Recommendation:
Promote this category through combo offers.
""")
    


# =========================================================
# SECTION 11: RATING ANALYSIS
# =========================================================

def rating_analysis(df):

    print(df["rating_given"].describe())
    plt.subplot(4,3,6)
   

    sns.histplot(
    df["rating_given"],
    kde=True,
    bins=10
)

    plt.title("Rating Distribution")

    plt.xlabel("Rating")

    plt.ylabel("Frequency")

    plt.grid(alpha=0.3)

    
    print("""Observation

Most ratings are concentrated between 2 and 3 stars.

Business Insight

Customers are generally not that much satisfied with the service.

Recommendation

Maintain food quality and delivery performance while addressing low-rated orders.""")


# =========================================================
# SECTION 12 : CORRELATION ANALYSIS
# =========================================================

def correlation_analysis(df):

    corr = (
        df.select_dtypes(
            include="number"
        )
        .corr()
    )

    print(corr.head())

    

   
    sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

    plt.title("Correlation Matrix")
    
    

    
    print("""Observation

Customer Total Spend shows a positive relationship with Customer Total Orders.

Business Insight

Customers who order more frequently tend to spend more overall.

Recommendation

Target high-frequency customers with premium memberships and special offers.""")


# =========================================================
# SECTION 13 : OUTLIER ANALYSIS
# =========================================================

def outlier_analysis(df):

    Q1 = df["order_value"].quantile(0.25)

    Q3 = df["order_value"].quantile(0.75)

    IQR = Q3 - Q1

    outliers = df[
        (df["order_value"] < Q1 - 1.5*IQR)
        |
        (df["order_value"] > Q3 + 1.5*IQR)
    ]

    print(
        f"Total Outliers: {len(outliers)}"
    )
    if len(outliers) == 0:
       print("No significant outliers found.")
    else:
       print(f"{len(outliers)} outliers detected.")
    print("""Observation

No orders have exceptionally high values.

Business Insight

These may represent smooth marketing.

Recommendation

Investigate high-value customers and design exclusive offers for them.""")
    
# =========================================================
# SECTION 14 : WEEKEND ANALYSIS
# =========================================================

def weekend_analysis(df):

    print("\n" + "=" * 60)
    print("DAY TYPE ANALYSIS")
    print("=" * 60)

    day_sales = (
        df.groupby("day_type")
        ["order_value"]
        .agg(["count","sum","mean"])
    )
    day_sales1 = (
        df.groupby("day_type")
        ["order_value"]
        .sum()
    )

    print(day_sales.head())
    best_day = day_sales["sum"].idxmax()

    print("\nINSIGHT:")
    print(
        f"{best_day} generates the highest sales."
    )
    plt.subplot(4,3,7)
    plt.pie(day_sales1.values,labels=day_sales1.index,autopct="%1.1f%%")
    plt.title("Revenue by Day Type")
    
    print("""Observation

Weekend sales are higher than weekday sales.

Business Insight

Customers are more likely to order food during weekends.

Recommendation

Launch weekend-exclusive offers and increase delivery capacity during peak periods.""")

# =========================================================
# SECTION 15 : Order Value Distribution
# =========================================================
def order_analysis(df):
    print("Average Order Value:",
      round(df["order_value"].mean(),2))
    print("Total Revenue:", df["order_value"].sum())
    plt.subplot(4,3,8)
    sns.histplot(df["order_value"],kde=True)
    plt.title("Order Value Distribution")  
    plt.xlabel("order values")
    plt.ylabel("frequency")

    print("""Observation

Most orders fall within the medium price range, while very high-value orders are less common.

Business Insight

The business relies heavily on regular-sized orders rather than luxury purchases.

Recommendation

Create combo offers to encourage customers to increase basket size.""")


# =========================================================
# SECTION 16 : Revenue by Mood
# =========================================================
def revenue_analysis(df):

   mood_sales=df.groupby("mood")["order_value"].sum().sort_index()
   plt.subplot(4,3,9)
   sns.barplot(x=mood_sales.index,y=mood_sales.values)
   plt.title("Revenue by Mood")
   plt.xlabel("mood")
   plt.ylabel("revenue")
   top_mood = mood_sales.idxmax()

   print(f"""
Observation:
Customers in '{top_mood}' mood generated the highest revenue.

Business Insight:
Customer mood influences ordering behavior.

Recommendation:
Target customers with personalized recommendations.
""")

# =========================================================
# SECTION 17 : Rainy Weather Impact
# =========================================================
def rainy_analysis(df):
   rain_sales = (
       df.groupby("rainy_weather")["order_value"]
       .sum()
   )

   plt.subplot(4,3,10)

   sns.barplot(
       x=rain_sales.index,
       y=rain_sales.values
   )

   plt.title("Sales During Rainy Weather")
   plt.xlabel("Rainy Weather")
   plt.ylabel("Revenue")


   best_weather = rain_sales.idxmax()

   print(f"""
Observation:
{best_weather} weather generated higher revenue.

Business Insight:
Weather impacts food ordering patterns.

Recommendation:
Launch weather-based promotions.
""")

# =========================================================
# SECTION 18 : Order Time Analysis
# =========================================================
def order_time_analysis(df):
   order_time=df.groupby("order_time")["order_value"].count().sort_index()
   plt.subplot(4,3,11)
   sns.barplot(
       x=order_time.index,
       y=order_time.values
   )

   plt.title("Orders by Time")

# =========================================================
# SECTION 19 : CUSTOMER SEGMENTS
# =========================================================
def customer_segment_analysis (df):
    customer_segment=df.groupby("Customer Segment")["order_value"].count().reset_index()
    print(customer_segment)
    plt.subplot(5,3,12)
    colors = [
    "red"
    if x == customer_segment["order_value"].max()
    else "skyblue"
    for x in customer_segment["order_value"]
]

    

    sns.barplot(
    x=customer_segment["Customer Segment"],
    y=customer_segment["order_value"],
    hue=customer_segment["Customer Segment"],
    palette=colors,
    legend=False
)

    for i, v in enumerate(customer_segment["order_value"]):

       plt.text(
        i,
        v,
        f"{v:,.0f}",
        ha="center"
    )

    plt.title("Revenue by Customer Segment")

    plt.xlabel("Customer Segment")

    plt.ylabel("Revenue")

    plt.grid(alpha=0.3)

    
    print("""Observation

Most customers belong to the VIP spending segments, while only a small percentage are Low or Medium customers.

Business Insight

Revenue growth can be achieved by focussing more on low and medium valued customers .

Recommendation

Use targeted promotions and personalized recommendations to increase spending.""")
# =========================================================
# Revenue by Restaurant Type
# =========================================================
def revenue_analysis(df):
   restaurant_sales = (
       df.groupby("restaurant_type")["order_value"]
         .sum()
         .sort_values(ascending=False)
   )



def dashboard(df):

    plt.figure(figsize=(20,12))

    city_analysis(df)
    customer_analysis(df)
    cuisine_analysis(df)
    delivery_analysis(df)
    meal_type_analysis(df)
    rating_analysis(df)
    weekend_analysis(df)
    order_analysis(df)
    rainy_analysis(df)
    order_time_analysis(df)
    customer_segment_analysis(df)

    plt.suptitle(
    "Food Ordering Business Dashboard",
    fontsize=22,
    fontweight="bold",
    y=1.02
)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
       r"images\business_dashboard.png",
       dpi=300,
       bbox_inches="tight"
    )

    plt.show()

def correlation_dashboard(df):

    corr = (
        df.select_dtypes(include="number")
        .corr()
    )

    plt.figure(figsize=(12,8))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Dashboard",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
       r"images\correlation_dashboard.png",
       dpi=300,
       bbox_inches="tight"
    )

    plt.show()

    

def executive_summary(df):

    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)

    print(
        f"Top City: "
        f"{df.groupby('city')['order_value'].sum().idxmax()}"
    )

    print(
        f"Top Cuisine: "
        f"{df.groupby('cuisine')['order_value'].sum().idxmax()}"
    )

    print(
        f"Best Meal Type: "
        f"{df.groupby('meal_type')['order_value'].sum().idxmax()}"
    )

    print(
        f"Repeat Customers: "
        f"{df['Repeat Customer'].sum()}"
    )

# =========================================================
# SECTION 20 : RUN PROJECT
# =========================================================
dataset_overview(df)

df = clean_data(df)
df = handle_missing_values(df)
df = feature_engineering(df)

dashboard(df)

correlation_dashboard(df)

outlier_analysis(df)

executive_summary(df)

print(df.head())
print("\nFinal Dataset Preview")
print(df.head())


