#import numpy as np
#import tensorflow

#a = np.random.random(10)
#print (a)


from flask import Flask
app = Flask(__name__)
@app.route('/')
def helloword():
    return 'Hello word!'

if __name__ == '__main__':
    app.debug = True
    app.run()