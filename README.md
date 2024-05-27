This python program uses machine learning to predict the intermediate of fungi sesquiterpene synthases(STS) and cluster fungi sesquiterpene synthases based on their sequence features.

Installation：
This program operates depending on the Python environment (Python Version 3.0 or above) and can be run on Windows systems.
Following python modules are needed:
	-- numpy
	-- matplotlib
	-- plotly
	-- sklearn
	-- xgboost
	-- pandas
	-- BioPython

parameters:
-i(required):A fasta file which contains all STS sequences for prediction
-o(required):A directory which all results and intermediate file will be stored
--C_domain(required):default:True, ML model will use the C domain of STS to predict. This will improve the accuarcy of prediction. But some STSs which lack complete C domain will be excluded.
--including(optional):default:False, the unsupervised clustering model will include more than 1000 STSs described in paper, we recommend you to add this parameter when you only have few STSs to predict. This may take a long time to make the alignment.
--without_supervised learning(optional):default:False. The supervised clustering model will predict the intermidiate of the input STSs, the outfile will store in prediction_result.txt.
--without_unsupervised learning(optional):default:False. The unsupervised clustering model will cluster the input STSs with all characterized (and uncharacterized)STSs. The cluster map will store in culstering_map.jpg and culstering_map.html. The coordinate of each STS will store in STS_coordinate.txt.

Usage:
python use_prediction.py -i test.fasta -o test

If you would like to include more than 1000 uncharacterized STSs described in paper in unsupervised clustering model:
python use_prediction.py -i test.fasta -o test --including
