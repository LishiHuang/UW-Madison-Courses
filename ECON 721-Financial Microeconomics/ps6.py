# %%
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.05  # Interest rate
ages = np.arange(20, 81)  # Ages from 20 to 80
T = len(ages)  # Total periods

# Scenario a)
# Savings s_t
s_t_a = np.zeros(T)
s_t_a[1:41] = 18665  # Ages 21-60
s_t_a[41:] = -181335  # Ages 61-80

# Wealth accumulation
w_t_a = np.zeros(T)
for t in range(1, T):
    w_t_a[t] = w_t_a[t-1] * (1 + r) + s_t_a[t]

# Scenario b)
# Inheritance at age 20
s_t_b = np.zeros(T)
s_t_b[0] = 1000000  # Inheritance at t=0
s_t_b[1:41] = -34128  # Ages 21-60
s_t_b[41:] = -234128  # Ages 61-80

# Wealth accumulation
w_t_b = np.zeros(T)
w_t_b[0] = s_t_b[0]
for t in range(1, T):
    w_t_b[t] = w_t_b[t-1] * (1 + r) + s_t_b[t]

# Scenario c)
# Bequest of $1,000,000 at age 80
s_t_c = np.zeros(T)
s_t_c[1:41] = 21518  # Ages 21-60
s_t_c[41:-1] = -178482  # Ages 61-79

# Wealth accumulation
w_t_c = np.zeros(T)
for t in range(1, T):
    w_t_c[t] = w_t_c[t-1] * (1 + r) + s_t_c[t]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(ages, w_t_a, label='Scenario a) No Inheritance')
plt.plot(ages, w_t_b, label='Scenario b) With Inheritance')
plt.plot(ages, w_t_c, label='Scenario c) Inheritance and Bequest')

plt.title('Wealth Accumulation Over a Lifetime')
plt.xlabel('Age')
plt.ylabel('Wealth ($)')
plt.legend()
plt.grid(True)
plt.show()


# %%

# Interest rates from 0.01% to 20%
r_values = np.linspace(0.0001, 0.20, 1000)  # Avoid r = 0
r_percent = r_values * 100  # Convert to percentage for plotting

# Exact doubling time
T_exact = np.log(2) / np.log(1 + r_values)

# Approximate doubling time (Rule of 72)
T_approx = 72 / r_percent

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(r_percent, T_exact, label='Exact Doubling Time')
plt.plot(r_percent, T_approx, label='Approximate Doubling Time (Rule of 72)', linestyle='--')
plt.title('Doubling Time vs. Interest Rate')
plt.xlabel('Interest Rate (%)')
plt.ylabel('Doubling Time (Years)')
plt.legend()
plt.grid(True)
plt.show()


# %%
import datetime
from dateutil.relativedelta import relativedelta

# Prompt the user for inputs
price = float(input("Enter the price of the house: "))
down_payment_percentage = float(input("Enter the down payment percentage (e.g., 10 for 10%): "))
annual_interest_rate = float(input("Enter the annual interest rate (e.g., 4 for 4%): "))

mortgage_years = eval(input("Select the mortgage maturity:"))

first_payment_date_str = input("Enter the date of the first payment (YYYY-MM-DD): ")

# Calculate the loan amount
down_payment = down_payment_percentage / 100 * price
loan_amount = price - down_payment

# Calculate the monthly interest rate
monthly_interest_rate = annual_interest_rate / 100 / 12

# Calculate the number of payments
number_of_payments = mortgage_years * 12

# Calculate the monthly payment using the annuity formula
x = loan_amount * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_payments) / \
    ((1 + monthly_interest_rate) ** number_of_payments - 1)

# Parse the first payment date
first_payment_date = datetime.datetime.strptime(first_payment_date_str, "%Y-%m-%d")

# Initialize lists to store the schedule
payment_dates = []
balances = []
interests = []
principal_paid = []

# Set the initial balance
balance = loan_amount
payment_date = first_payment_date

# Generate the payment schedule
for i in range(int(number_of_payments)):
    interest = balance * monthly_interest_rate
    principal = x - interest
    balance -= principal
    payment_dates.append(payment_date)
    balances.append(balance if balance > 0 else 0)
    interests.append(interest)
    principal_paid.append(principal)
    # Move to the next payment date
    payment_date += relativedelta(months=+1)

# Get the end payment date
end_payment_date = payment_dates[-1]

# Print the result
print(f"\nYour monthly payment between {first_payment_date.strftime('%Y-%m-%d')} and {end_payment_date.strftime('%Y-%m-%d')} is ${x:.2f}.")

# Plotting the remaining balance over time
plt.figure(figsize=(12, 6))
plt.plot(payment_dates, balances, label='Remaining Balance')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.title('Remaining Mortgage Balance Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the interest paid over time
plt.figure(figsize=(12, 6))
plt.plot(payment_dates, interests, label='Interest Paid Each Month', color='orange')
plt.xlabel('Date')
plt.ylabel('Amount ($)')
plt.title('Monthly Interest Paid Over Time')
plt.legend()
plt.grid(True)
plt.show()



