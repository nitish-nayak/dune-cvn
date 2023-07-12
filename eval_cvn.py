from cvn_utils import *

pm = get_pixelmap('3', 'nue')
info = get_eventinfo('3', 'nue')
model = get_model() # get the neural network model

#  print(get_pmresults(pm, info, 1, model))
result = {'px': -1, 'py': -1}
nom_pmresult = get_pmresults(pm, info, 1, model)
result.update(nom_pmresult)
results = [result]

for ix in range(500):
    for iy in range(500):
        temp_res = {'px':ix, 'py':iy}
        
        check1 = (pm[0][ix][iy] == 0)
        check2 = (pm[1][ix][iy] == 0)
        check3 = (pm[2][ix][iy] == 0)
        if check1 and check2 and check3:
            temp_res.update(nom_pmresult)
        else:
            temp_res.update(turnoffPixel(ix, iy, pm, info, 1, model))
        
        results.append(temp_res)

df = pd.DataFrame(results)
print(df['nue_score'].unique())
