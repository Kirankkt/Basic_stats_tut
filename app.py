import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# Set the page configuration
st.set_page_config(page_title="Statistics Tutorial", layout="wide")

# Sidebar navigation
st.sidebar.title("Statistics Tutorial")
options = st.sidebar.radio("Select a topic:", [
    "Introduction",
    "Measures of Central Tendency",
    "Probability Density Functions (PDFs)",
    "Distributions",
    "Hypothesis Testing",
    "Inferential Statistics",
    "Regression"
])

# Introduction Page
if options == "Introduction":
    st.title("Welcome to the Statistics Tutorial")
    st.write("""
    This application is designed to help you understand fundamental concepts in statistics. 
    Navigate through the sidebar to explore different topics, complete with explanations and interactive visualizations.
    """)

# Measures of Central Tendency
elif options == "Measures of Central Tendency":
    st.title("Measures of Central Tendency")
    st.write("""
    Measures of central tendency describe the center point or typical value of a dataset. The main measures are:
    - **Mean:** The average of all data points.
    - **Median:** The middle value when data points are ordered.
    - **Mode:** The most frequently occurring data point.
    """)
    
    # Generate sample data
    data = np.random.normal(loc=50, scale=15, size=1000)
    
    mean = np.mean(data)
    median = np.median(data)
    mode = stats.mode(data)[0][0]
    
    st.subheader("Sample Data Visualization")
    fig, ax = plt.subplots()
    sns.histplot(data, bins=30, kde=True, ax=ax)
    ax.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
    ax.axvline(median, color='g', linestyle='-', label=f'Median: {median:.2f}')
    ax.axvline(mode, color='b', linestyle=':', label=f'Mode: {mode:.2f}')
    ax.legend()
    st.pyplot(fig)
    
    st.write(f"**Mean:** {mean:.2f}")
    st.write(f"**Median:** {median:.2f}")
    st.write(f"**Mode:** {mode:.2f}")

# Probability Density Functions (PDFs)
elif options == "Probability Density Functions (PDFs)":
    st.title("Probability Density Functions (PDFs)")
    st.write("""
    A Probability Density Function describes the likelihood of a continuous random variable to take on a particular value.
    The area under the curve of a PDF equals 1.
    """)
    
    st.subheader("Normal Distribution PDF")
    mean = st.slider("Select Mean", 0.0, 100.0, 50.0)
    std = st.slider("Select Standard Deviation", 1.0, 30.0, 15.0)
    
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = stats.norm.pdf(x, mean, std)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Normal PDF')
    ax.fill_between(x, y, alpha=0.2)
    ax.set_title('Normal Distribution PDF')
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability Density')
    ax.legend()
    st.pyplot(fig)

# Distributions
elif options == "Distributions":
    st.title("Distributions")
    st.write("""
    Distributions describe how values of a random variable are spread or distributed. Common distributions include:
    - **Normal Distribution**
    - **Binomial Distribution**
    - **Poisson Distribution**
    - **Uniform Distribution**
    """)
    
    distribution = st.selectbox("Select a distribution to visualize:", [
        "Normal",
        "Binomial",
        "Poisson",
        "Uniform"
    ])
    
    if distribution == "Normal":
        mean = st.slider("Mean", -10.0, 10.0, 0.0)
        std = st.slider("Standard Deviation", 0.1, 5.0, 1.0)
        x = np.linspace(mean - 4*std, mean + 4*std, 1000)
        y = stats.norm.pdf(x, mean, std)
        title = f'Normal Distribution (μ={mean}, σ={std})'
    
    elif distribution == "Binomial":
        n = st.slider("Number of trials (n)", 1, 50, 10)
        p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5)
        x = np.arange(0, n+1)
        y = stats.binom.pmf(x, n, p)
        title = f'Binomial Distribution (n={n}, p={p})'
    
    elif distribution == "Poisson":
        mu = st.slider("Rate parameter (λ)", 0.1, 10.0, 3.0)
        x = np.arange(0, mu + 10)
        y = stats.poisson.pmf(x, mu)
        title = f'Poisson Distribution (λ={mu})'
    
    elif distribution == "Uniform":
        a = st.slider("Lower bound (a)", -10.0, 0.0, -5.0)
        b = st.slider("Upper bound (b)", 0.0, 20.0, 5.0)
        x = np.linspace(a, b, 1000)
        y = stats.uniform.pdf(x, loc=a, scale=b-a)
        title = f'Uniform Distribution (a={a}, b={b})'
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(x, y, label=distribution)
    ax.fill_between(x, y, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability')
    ax.legend()
    st.pyplot(fig)

# Hypothesis Testing
elif options == "Hypothesis Testing":
    st.title("Hypothesis Testing")
    st.write("""
    Hypothesis testing is a method for testing a claim or hypothesis about a parameter in a population, using data measured in a sample.
    Steps involved:
    1. State the null and alternative hypotheses.
    2. Choose the significance level (α).
    3. Compute the test statistic.
    4. Make a decision based on the p-value or critical value.
    """)
    
    st.subheader("One-Sample t-Test Example")
    # Generate sample data
    data = np.random.normal(loc=50, scale=10, size=100)
    st.write("**Sample Data:**")
    st.write(pd.Series(data).describe())
    
    # Hypothesis
    st.write("""
    **Null Hypothesis (H₀):** μ = 50  
    **Alternative Hypothesis (H₁):** μ ≠ 50
    """)
    
    # Perform t-test
    t_stat, p_value = stats.ttest_1samp(data, 50)
    
    st.write(f"**t-Statistic:** {t_stat:.4f}")
    st.write(f"**p-Value:** {p_value:.4f}")
    
    alpha = st.slider("Select Significance Level (α)", 0.01, 0.10, 0.05, step=0.01)
    if p_value < alpha:
        st.write(f"**Decision:** Reject the null hypothesis at α = {alpha}")
    else:
        st.write(f"**Decision:** Fail to reject the null hypothesis at α = {alpha}")

# Inferential Statistics
elif options == "Inferential Statistics":
    st.title("Inferential Statistics")
    st.write("""
    Inferential statistics allow us to make inferences about a population based on a sample of data. Key concepts include:
    - **Confidence Intervals**
    - **Significance Levels**
    - **p-Values**
    """)
    
    st.subheader("Confidence Interval for Mean")
    # Generate sample data
    data = np.random.normal(loc=100, scale=20, size=200)
    mean = np.mean(data)
    sem = stats.sem(data)
    confidence = st.slider("Select Confidence Level (%)", 80, 99, 95, step=1)
    ci = stats.t.interval(confidence / 100, len(data)-1, loc=mean, scale=sem)
    
    st.write(f"**Sample Mean:** {mean:.2f}")
    st.write(f"**Confidence Interval ({confidence}%):** [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    # Plot
    fig, ax = plt.subplots()
    sns.histplot(data, bins=30, kde=True, ax=ax)
    ax.axvline(ci[0], color='r', linestyle='--', label=f'Lower CI: {ci[0]:.2f}')
    ax.axvline(ci[1], color='g', linestyle='--', label=f'Upper CI: {ci[1]:.2f}')
    ax.axvline(mean, color='b', linestyle='-', label=f'Mean: {mean:.2f}')
    ax.legend()
    st.pyplot(fig)

# Regression
elif options == "Regression":
    st.title("Regression Analysis")
    st.write("""
    Regression analysis estimates the relationships among variables. The most common type is linear regression.
    """)
    
    st.subheader("Simple Linear Regression Example")
    # Generate synthetic data
    np.random.seed(0)
    x = np.random.rand(100) * 10
    y = 2.5 * x + np.random.randn(100) * 5
    df = pd.DataFrame({'X': x, 'Y': y})
    
    # Plot data
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='X', y='Y', ax=ax)
    st.pyplot(fig)
    
    # Perform regression
    X = sm.add_constant(df['X'])
    model = sm.OLS(df['Y'], X).fit()
    predictions = model.predict(X)
    
    st.write(model.summary())
    
    # Plot regression line
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='X', y='Y', ax=ax, label='Data Points')
    sns.lineplot(x=df['X'], y=predictions, color='r', label='Regression Line', ax=ax)
    ax.set_title('Simple Linear Regression')
    ax.legend()
    st.pyplot(fig)
