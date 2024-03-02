import pandas as pd 

def test_series():
    sr = pd.Series([17000, 18000, 1000, 5000], index=["피자", "치킨", "콜라", "맥주"])
    print("Series")
    print('-'*15)
    print(sr)

    print(f'values: {sr.values}')
    print(f'index:  {sr.index}')



def test_data_frame():
    values = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]
    index = ['one', 'two', 'three']
    columns = ['A', 'B', 'C']

    df = pd.DataFrame(values, index=index, columns=columns)
    print('DataFrame')
    print('-'*18)
    print(df)

    # From list
    print("* data frame from list")
    data = [
        ['1000', 'Steve', 90.72], 
        ['1001', 'James', 78.09], 
        ['1002', 'Doyeon', 98.43], 
        ['1003', 'Jane', 64.19], 
        ['1004', 'Pilwoong', 81.30],
        ['1005', 'Tony', 99.14],
    ]
    df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
    print(df)
    x = df['학번']
    y = df['이름']
    print('x: ', x.to_list())
    print('y: ', y.to_list())    

    # From dictionary
    print("*  data frame from dictionary")
    data = {
        '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
        '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
        '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }
    df = pd.DataFrame(data, columns=['학번', '이름', '점수'])
    print(df)
    print(df.head(3))
    print(df.tail(3))
    print(df['학번'])

    print("* data frame from dictionary - no column parameter")
    df = pd.DataFrame(data)
    print(df)



#test_series()
test_data_frame()
