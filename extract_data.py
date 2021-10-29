import numpy as np
import pandas as pd

def create_dataframe():
    columns = ['Title', 'Brand Name', 'Material', 'Color', 'Category']
    with open("data/publish_data.txt") as f:
        lines = f.readlines()
    my_list = []
    for line in lines:
        inside_list = [None, None, None, None, None]
        data = line.rstrip().split('\x01')
        if data[2] == 'NULL':
            continue
        else:
            try:
                j = columns.index(data[1])
                inside_list[0] = data[0].lstrip().lower()
                inside_list[j] = data[2].lower()
            except:
                continue
        my_list.append(inside_list)
    
    df = pd.DataFrame(my_list, columns=columns)
    
    return df


def arrange_dataframe():
    columns = ['Title', 'Brand Name', 'Material', 'Color', 'Category']
    df = create_dataframe()
    test_df = df.groupby(df.Title.values).agg(lambda x: list(filter(None, x.tolist())))
    my_list = []
    for index, row in test_df.iterrows():
        inside_list = []
        for column in columns:
            try:
                inside_list.append(row[column][0])
            except:
                inside_list.append(None)
        
        my_list.append(inside_list)
    
    my_df = pd.DataFrame(data=my_list, columns=columns)

    return my_df

def main():
    df = arrange_dataframe()
    df.to_csv('data/published_data.csv', index=False)

if __name__ == '__main__':
    main()
