data = [
    {'bbox': [401, 144, 451, 202], 'class': 2, 'confidence': 0.6260728240013123, 'obj_name': 'car'},
    {'bbox': [356, 73, 393, 111], 'class': 2, 'confidence': 0.5421959757804871, 'obj_name': 'car'},
    {'bbox': [567, 190, 628, 265], 'class': 2, 'confidence': 0.5160531401634216, 'obj_name': 'car'}
]

converted_data = [(item['bbox'], item['confidence'], item['obj_name']) for item in data]

print(converted_data)
