
import joblib


results_path={
    "housegan":"housegan_experiments.pkl",
    "mnist":"mnist_experiments.pkl",
    "cifar10":"cifar10_experiments.pkl",
    "drug":"drug_experiments.pkl"
}

def get_results(model_name):
    two_results=joblib.load(results_path.get(model_name))
    import math
    for key in two_results.keys():
        print(key)
        results=two_results[key]

        for j in range(len(results)):
            new_results=[]
            result=results[j]
            for i in range(len(result)):
                results1=round(result[i],1)
                new_results.append(results1)
            print("{}({}) {}({})".format(new_results[0],round(math.sqrt(new_results[1]),1),new_results[2],
            round(math.sqrt(new_results[3]),1)))

def parese_results():
    file_path="line_graph.pkl"
    results=joblib.load(file_path)

    for j in range(len(results)):
        new_results=[]
        result=results[j]
        for i in range(len(result)):
            results1=round(result[i],1)
            new_results.append(results1)
        print("{} {}".format(new_results[0],new_results[2]))


if __name__=="__main__":
    # get_results("mnist")
    # get_results("mnist")
    parese_results()

    pass