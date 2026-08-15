#Libraries + CSV Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
print(df)

df.head()
df.info()
df.describe()

#CustomerID remove + Age visualization
df.drop(["CustomerID"], axis=1, inplace=True)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

plt.figure(figsize=(10, 6))
plt.title("Ages Frequency")
sns.axes_style("dark")
sns.violinplot(y=df["Age"])
plt.show()

#Spending Score + Annual Income Boxplot
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.boxplot(y=df["Spending Score (1-100)"], color="red")

plt.subplot(1, 2, 2)
sns.boxplot(y=df["Annual Income (k$)"])

plt.show()


#Gender Analysis
genders = df.Gender.value_counts()

sns.set_style("darkgrid")

plt.figure(figsize=(10, 4))
sns.barplot(x=genders.index, y=genders.values)

plt.show()

#Age Groups
age18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age55above = df.Age[df.Age >= 56]

x = ["18-25", "26-35", "36-45", "46-55", "55+"]
y = [
    len(age18_25.values),
    len(age26_35.values),
    len(age36_45.values),
    len(age46_55.values),
    len(age55above.values)
]

plt.figure(figsize=(15, 6))
sns.barplot(x=x, y=y, palette="rocket")

plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")

plt.show()

#Spending Score Groups
ss1_20 = df["Spending Score (1-100)"][
    (df["Spending Score (1-100)"] >= 1) &
    (df["Spending Score (1-100)"] <= 20)
]

ss21_40 = df["Spending Score (1-100)"][
    (df["Spending Score (1-100)"] >= 21) &
    (df["Spending Score (1-100)"] <= 40)
]

ss41_60 = df["Spending Score (1-100)"][
    (df["Spending Score (1-100)"] >= 41) &
    (df["Spending Score (1-100)"] <= 60)
]

ss61_80 = df["Spending Score (1-100)"][
    (df["Spending Score (1-100)"] >= 61) &
    (df["Spending Score (1-100)"] <= 80)
]

ss81_100 = df["Spending Score (1-100)"][
    (df["Spending Score (1-100)"] >= 81) &
    (df["Spending Score (1-100)"] <= 100)
]

ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]

ssy = [
    len(ss1_20.values),
    len(ss21_40.values),
    len(ss41_60.values),
    len(ss61_80.values),
    len(ss81_100.values)
]

plt.figure(figsize=(15, 6))
sns.barplot(x=ssx, y=ssy, palette="nipy_spectral_r")

plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")

plt.show()

#Annual Income Groups
ai0_30 = df["Annual Income (k$)"][
    (df["Annual Income (k$)"] >= 0) &
    (df["Annual Income (k$)"] <= 30)
]

ai31_60 = df["Annual Income (k$)"][
    (df["Annual Income (k$)"] >= 31) &
    (df["Annual Income (k$)"] <= 60)
]

ai61_90 = df["Annual Income (k$)"][
    (df["Annual Income (k$)"] >= 61) &
    (df["Annual Income (k$)"] <= 90)
]

ai91_120 = df["Annual Income (k$)"][
    (df["Annual Income (k$)"] >= 91) &
    (df["Annual Income (k$)"] <= 120)
]

ai121_150 = df["Annual Income (k$)"][
    (df["Annual Income (k$)"] >= 121) &
    (df["Annual Income (k$)"] <= 150)
]

aix = [
    "$ 0 - 30,000",
    "$ 30,001 - 60,000",
    "$ 60,001 - 90,000",
    "$ 90,001 - 120,000",
    "$ 120,001 - 150,000"
]

aiy = [
    len(ai0_30.values),
    len(ai31_60.values),
    len(ai61_90.values),
    len(ai91_120.values),
    len(ai121_150.values)
]

plt.figure(figsize=(15, 6))
sns.barplot(x=aix, y=aiy, palette="Set2")

plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")

plt.show()

#K-Means — Elbow Method
X = df[["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]].copy()

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(
    range(1, 11),
    wcss,
    linewidth=2,
    color="red",
    marker="8"
)
plt.xlabel("K Value")
plt.xticks(np.arange(1, 11, 1))
plt.ylabel("WCSS")
plt.show()

# Fit final model and add cluster labels
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
df["label"] = kmeans.fit_predict(X)

#3D Customer Segmentation Output
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df.Age[df.label == 0],
    df["Annual Income (k$)"][df.label == 0],
    df["Spending Score (1-100)"][df.label == 0],
    c='blue',
    s=60
)

ax.scatter(
    df.Age[df.label == 1],
    df["Annual Income (k$)"][df.label == 1],
    df["Spending Score (1-100)"][df.label == 1],
    c='red',
    s=60
)

ax.scatter(
    df.Age[df.label == 2],
    df["Annual Income (k$)"][df.label == 2],
    df["Spending Score (1-100)"][df.label == 2],
    c='green',
    s=60
)

ax.scatter(
    df.Age[df.label == 3],
    df["Annual Income (k$)"][df.label == 3],
    df["Spending Score (1-100)"][df.label == 3],
    c='orange',
    s=60
)

ax.scatter(
    df.Age[df.label == 4],
    df["Annual Income (k$)"][df.label == 4],
    df["Spending Score (1-100)"][df.label == 4],
    c='purple',
    s=60
)

ax.view_init(30, 185)

plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")

plt.show()


# for run use this:- python .\customerseg.py