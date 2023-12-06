from flask import Flask, request

app = Flask(__name__)

# Dictionary of products and their weights
products = {
    'ProdA': 71.3,
    'ProdB': 15.0,
    'ProdC': 4.4
}

@app.route('/determine_change/', methods=['POST'])
def receive_weight_change():

    '''
    A method that retrieves received weight values and verifies their presence in the dictionary.
    '''
    data = request.get_json()
    
    sensor_values = data.get('sensor_values')

    if sensor_values:
        weight_sum = sum(sensor_values)

        product_name = find_product_by_weight(weight_sum)
        if product_name:
            message = f"Product found: {product_name}, weight: {weight_sum}"
            print(message)
            return message, 200
        else:
            message = "Error: Product not found for the given weight."
            print(message)
            return message, 404

def find_product_by_weight(weight, thrs=1):
    for name, product_weight in products.items():
        if abs(weight - product_weight) < thrs: 
            return name
        
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0')