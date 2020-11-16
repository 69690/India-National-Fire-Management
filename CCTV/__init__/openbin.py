import pickle

f = open('cache.dat','rb')
try:
    while True:
        obj = pickle.load(f)
        print(obj.cctv_id)
        print(obj.location)
        print(obj.date)
        print(obj.time)
except EOFError:
    pass