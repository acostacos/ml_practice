import pandas as pd
import matplotlib.pyplot as plt
from gradient_descent import train_model


def main():
    # Given total population predict urban population for a specific country
    FILENAME = "urban_population_analysis_1950_to_2050.csv"
    COUNTRY = 'Philippines'
    INIT_W = 0
    INIT_B = 0
    ALPHA = 0.00006
    ITERATIONS = 10500

    # Get Data
    print('Retrieving data')
    filepath = f"../../data/{FILENAME}"
    df = pd.read_csv(filepath)
    df = df[df['Economy Label'] == COUNTRY]

    # Clean Data
    print('Cleaning data')
    df = df[['Absolute value in thousands',
             'Urban population as percentage of total population']]
    column_names = {
        'Absolute value in thousands': 'x',
        'Urban population as percentage of total population': 'y',
    }
    df = df.rename(column_names, axis='columns')
    df = df[(df['x'] != 'NOT APPLICABLE') & (df['y'] != 'NOT APPLICABLE')]
    df[['x', 'y']] = df[['x', 'y']].apply(lambda x: x.astype('float'))

    # Feature scaling
    max_x = df['x'].max()
    df['x'] = df['x'] / max_x

    # Show data before Gradient Descent
    # df.plot(x='x', y='y', kind='scatter')
    # plt.show()

    # Perform Gradient Descent
    print('Computing for parameters')
    w, b = train_model(df, INIT_W, INIT_B, ALPHA, ITERATIONS)

    # w = 1.2792182750724967
    # b = 2.228122168426517

    print(f'w: {w}')
    print(f'b: {b}')

    # Show data after Gradient Descent with best fit line
    df.plot(x='x', y='y', kind='scatter')
    bfl_x = [0.1, 0.5, 1]
    bfl_y = [(w * x) + b for x in bfl_x]
    print(bfl_x)
    print(bfl_y)
    plt.plot(bfl_x, bfl_y)
    plt.show()


if __name__ == "__main__":
    main()
