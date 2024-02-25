import numpy as np
import pandas as pd


if __name__ == '__main__':
    q_table = pd.DataFrame(columns=list(range(4)), dtype=np.float64)
    print(q_table.columns)
    q_table = q_table._append(
        pd.Series(
            [0] * 4,
            index=q_table.columns,
            name='[0,0]',
        )
    )
    print(q_table)
