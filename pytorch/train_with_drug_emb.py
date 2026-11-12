import os
import argparse
import random
import logging
from warnings import simplefilter

import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning

from data_processor.datasets import SubgraphDataset
from data_processor.subgraph_extraction import generate_subgraph_datasets
from manager.evaluator import Evaluator_multiclass, Evaluator_multilabel
from manager.trainer import Trainer
from model.Classifier_model_with_drug_emb import ClassifierModelWithDrugEmb
from utils.initialization_utils import initialize_experiment

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def _read_drug_embedding_metadata(embeddings_path):
    loaded = torch.load(embeddings_path, map_location='cpu')
    if torch.is_tensor(loaded):
        if loaded.dim() != 2:
            raise ValueError('Drug embedding matrix must be 2D: (num_drugs, embedding_dim).')
        return loaded.shape[0], loaded.shape[1]

    if not isinstance(loaded, dict) or 'drug_embeddings' not in loaded:
        raise ValueError("Embedding file must be a tensor or a dict containing 'drug_embeddings'.")

    drug_embeddings = loaded['drug_embeddings']
    if drug_embeddings.dim() != 2:
        raise ValueError('drug_embeddings must be 2D: (num_drugs, embedding_dim).')

    num_drug_nodes = loaded.get('num_drug_nodes', drug_embeddings.shape[0])
    emb_dim = loaded.get('embedding_dim', drug_embeddings.shape[1])
    return int(num_drug_nodes), int(emb_dim)


def initialize_model_with_drug_emb(params, model_class):
    checkpoint_path = os.path.join(params.exp_dir, 'best_graph_classifier.pth')
    if params.load_model and os.path.exists(checkpoint_path):
        logging.info('Loading existing model from %s', checkpoint_path)
        classifier = torch.load(checkpoint_path).to(device=params.device)
    else:
        logging.info('Initializing new model with external drug embeddings.')
        classifier = model_class(params).to(device=params.device)

        if params.use_drug_embeddings and params.drug_embeddings_path:
            logging.info('Loading drug embeddings from %s', params.drug_embeddings_path)
            classifier.load_drug_embeddings(params.drug_embeddings_path)
            classifier = classifier.to(device=params.device)
        else:
            logging.info('Drug embedding path not provided; using learnable embeddings for all nodes.')
            classifier.embedding_model.use_drug_embeddings = False

    return classifier


def process_dataset(params):
    params.db_path = os.path.join(
        params.main_dir, f'../data/{params.dataset}/digraph_hop_{params.hop}_{params.BKG_file_name}'
    )

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train_data = SubgraphDataset(
        db_path=params.db_path,
        db_name='train_subgraph',
        raw_data_paths=params.file_paths,
        add_traspose_rels=params.add_traspose_rels,
        use_pre_embeddings=params.use_pre_embeddings,
        dataset=params.dataset,
        kge_model=params.kge_model,
        dig_layer=params.num_dig_layers,
        BKG_file_name=params.BKG_file_name,
    )

    test_data = SubgraphDataset(
        db_path=params.db_path,
        db_name='test_subgraph',
        use_pre_embeddings=params.use_pre_embeddings,
        dataset=params.dataset,
        kge_model=params.kge_model,
        ssp_graph=train_data.ssp_graph,
        id2entity=train_data.id2entity,
        id2relation=train_data.id2relation,
        rel=train_data.num_rels,
        global_graph=train_data.global_graph,
        dig_layer=params.num_dig_layers,
        BKG_file_name=params.BKG_file_name,
    )

    valid_data = SubgraphDataset(
        db_path=params.db_path,
        db_name='valid_subgraph',
        use_pre_embeddings=params.use_pre_embeddings,
        dataset=params.dataset,
        kge_model=params.kge_model,
        ssp_graph=train_data.ssp_graph,
        id2entity=train_data.id2entity,
        id2relation=train_data.id2relation,
        rel=train_data.num_rels,
        global_graph=train_data.global_graph,
        dig_layer=params.num_dig_layers,
        BKG_file_name=params.BKG_file_name,
    )

    params.num_rels = train_data.num_rels
    params.global_graph = train_data.global_graph.to(params.device)
    params.aug_num_rels = train_data.aug_num_rels
    params.num_nodes = 35000

    if params.use_drug_embeddings and params.drug_embeddings_path:
        params.num_drug_nodes, params.drug_emb_input_dim = _read_drug_embedding_metadata(
            params.drug_embeddings_path
        )
        logging.info(
            'Drug embedding info: %d nodes with %d-dim input features',
            params.num_drug_nodes,
            params.drug_emb_input_dim,
        )

    logging.info("Device: %s", params.device)
    logging.info(" # Relations : %s, # Augmented relations : %s", params.num_rels, params.aug_num_rels)
    return train_data, valid_data, test_data


def main(params):
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file)),
        'test': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.test_file)),
    }

    train_data, valid_data, test_data = process_dataset(params)
    classifier = initialize_model_with_drug_emb(params, ClassifierModelWithDrugEmb)

    valid_evaluator = Evaluator_multiclass(params, classifier, valid_data) if params.dataset == 'drugbank' \
        else Evaluator_multilabel(params, classifier, valid_data)
    test_evaluator = Evaluator_multiclass(params, classifier, test_data, is_test=True) if params.dataset == 'drugbank' \
        else Evaluator_multilabel(params, classifier, test_data)
    print(classifier)

    trainer = Trainer(params, classifier, train_data, valid_evaluator, test_evaluator)
    logging.info('start training...')
    trainer.train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='KnowDDI with external drug embeddings')

    parser.add_argument('--seed', type=int, default=1111, help='seeds for random initial')
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true', help='Load existing model?')
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="Drugbank_DrugEmb",
        help="A folder with this name would be created to dump saved models and log files",
    )

    parser.add_argument('--dataset', "-d", type=str, default='drugbank')
    parser.add_argument("--train_file", "-tf", type=str, default="train", help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid", help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test", help="Name of file containing validation triplets")
    parser.add_argument("--kge_model", type=str, default="TransE", help="Which KGE model to load entity embeddings from")
    parser.add_argument("--use_pre_embeddings", type=bool, default=False, help='whether to use pretrained KGE embeddings')
    parser.add_argument('--BKG_file_name', type=str, default='BKG_file')

    parser.add_argument(
        "--drug_embeddings_path",
        type=str,
        default="../data/drugbank/drug_embeddings.pt",
        help="Path to an external embedding matrix or dict containing 'drug_embeddings'",
    )
    parser.add_argument("--use_drug_embeddings", type=bool, default=True, help="Whether to use external drug embeddings")
    parser.add_argument("--drug_emb_hidden_dim", type=int, default=512, help="Hidden dimension for the drug embedding projector")
    parser.add_argument("--drug_emb_dropout", type=float, default=0.1, help="Dropout for the drug embedding projector")

    parser.add_argument("--max_links", type=int, default=250000, help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=200, help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--enclosing_subgraph', '-en', type=bool, default=True, help='whether to only consider enclosing subgraph')
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False, help='whether to append adj matrix list with symmetric relations')

    parser.add_argument("--eval_every_iter", type=int, default=526, help="Interval of iterations to evaluate the model")
    parser.add_argument("--save_every_epoch", type=int, default=10, help="Interval of epochs to save a checkpoint of the model")
    parser.add_argument("--early_stop_epoch", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate of the optimizer")
    parser.add_argument("--lr_decay_rate", type=float, default=0.93, help="adjust the learning rate via epochs")
    parser.add_argument("--weight_decay_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", "-ne", type=int, default=50, help="numer of epochs")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of dataloading processes")

    parser.add_argument("--emb_dim", "-dim", type=int, default=32, help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", type=int, default=2, help="Number of GCN layers")
    parser.add_argument('--gcn_aggregator_type', type=str, choices=['mean', 'gcn', 'pool'], default='mean')
    parser.add_argument("--gcn_dropout", type=float, default=0.2, help="node_dropout rate in GCN layers")

    parser.add_argument("--num_infer_layers", type=int, default=3, help="Number of infer layers")
    parser.add_argument("--num_dig_layers", type=int, default=3)
    parser.add_argument("--MLP_hidden_dim", type=int, default=16)
    parser.add_argument("--MLP_num_layers", type=int, default=2)
    parser.add_argument("--MLP_dropout", type=float, default=0.2)
    parser.add_argument("--func_num", type=int, default=1)
    parser.add_argument("--sparsify", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--edge_softmax", type=int, default=1)
    parser.add_argument("--gsl_rel_emb_dim", type=int, default=32)
    parser.add_argument("--lamda", type=float, default=0.7)
    parser.add_argument("--gsl_has_edge_emb", type=int, default=1)

    params = parser.parse_args()

    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    set_seed(params.seed)
    initialize_experiment(params, __file__)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    main(params)
