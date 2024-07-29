import obonet
import networkx as nx
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter, defaultdict
from joblib import Parallel, delayed
import tqdm
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import logging
from typing import Dict, List, Tuple, Any
import os
from dotenv import load_dotenv
import argparse
import configparser
import sys 

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = './config.ini'

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def parse_hpo_ontology(file_path: str) -> Tuple[nx.Graph, Dict[str, Dict[str, Any]]]:
    """
    Parse the HPO ontology from an OBO file

    Args:
        file_path: Path to the HPO OBO file

    Returns:
        A tuple containing the HPO ontology graph and term information dictionary
    """
    try:
        graph = obonet.read_obo(file_path)
    except FileNotFoundError:
        logger.error(f"HPO ontology file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsinHPO ontology: {str(e)}")
        raise

    term_info = {}
    for node, data in graph.nodes(data=True):
        synonyms = [s.split('"')[1] for s in data.get('synonym', [])]
        definition = data.get('def', [''])[0].split('"')[1] if 'def' in data else ''
        term_info[node] = {
            'name': data.get('name', ''),
            'synonyms': synonyms,
            'definition': definition
        }
    
    return graph, term_info

def process_hpoa_annotations(file_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Process HPOA annotations 

    Args:
        file_path: Path to the HPOA annotations file

    Returns:
        A tuple containing mappings of diseases to HPO terms
    """
    disease_to_hpo = defaultdict(set)
    hpo_to_disease = defaultdict(set)
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:
                    database_id, hpo_id = row[0], row[3]
                    disease_to_hpo[database_id].add(hpo_id)
                    hpo_to_disease[hpo_id].add(database_id)
    except FileNotFoundError:
        logger.error(f"HPOA annotations file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing HPOA annotations: {str(e)}")
        raise
    return dict(disease_to_hpo), dict(hpo_to_disease)

def process_patient_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load patient data 

    Args:
        file_path: Path to the patient data JSON file

    Returns:
        A list of patient data dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Patient data file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in patient data file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error processing patient data: {str(e)}")
        raise

def expand_hpos(patient: Dict[str, Any], hpo_graph: nx.Graph) -> List[str]:
    """
    Expand a patient's HPO terms to include ancestors

    Args:
        patient: A patient data dictionary
        hpo_graph: The HPO ontology graph

    Returns:
        A list of expanded HPO terms
    """
    expanded_hpos = set()
    for hpo in patient['hpos']:
        if hpo in hpo_graph:
            expanded_hpos.add(hpo)
            expanded_hpos.update(nx.ancestors(hpo_graph, hpo))
        else:
            logger.warning(f"HPO term {hpo} not found in the ontology graph")
    return list(expanded_hpos)

def connect_data(patients: List[Dict[str, Any]], hpo_graph: nx.Graph, 
                 disease_to_hpo: Dict[str, List[str]], hpo_to_disease: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Connect patient data with HPO and disease information

    Args:
        patients: List of patient data dictionaries
        hpo_graph: The HPO ontology graph
        disease_to_hpo: Mapping of diseases to HPO
        hpo_to_disease: Mapping of HPO terms to disease

    Returns:
        Updated list of patient data dictionaries
    """
    expanded_hpos_list = Parallel(n_jobs=-1)(
        delayed(expand_hpos)(patient, hpo_graph) for patient in tqdm.tqdm(patients, desc="Expanding HPOs")
    )
    
    for patient, expanded_hpos in zip(patients, expanded_hpos_list):
        patient['expanded_hpos'] = expanded_hpos
        
        potential_diseases = set()
        for hpo in expanded_hpos:
            potential_diseases.update(hpo_to_disease.get(hpo, []))
        patient['potential_diseases'] = list(potential_diseases)
    
    return patients

def calculate_ic(disease_to_hpo: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate the information content of HPO terms

    Args:
        disease_to_hpo: Mapping of diseases to HPO terms

    Returns:
        Information content for each HPO term
    """
    term_freq = Counter()
    for hpos in disease_to_hpo.values():
        term_freq.update(hpos)
    
    total_diseases = len(disease_to_hpo)
    ic = {term: -np.log((count + 1) / (total_diseases + 1)) for term, count in term_freq.items()}
    return ic

def create_weighted_hpo_features(patients: List[Dict[str, Any]], ic: Dict[str, float], max_features: int) -> Tuple[np.ndarray, List[str]]:
    """
    Create weighted HPO features for patients

    Args:
        patients: List of patient data dictionaries
        ic: Information content for each HPO term
        max_features: Maximum number of features to include

    Returns:
        Feature matrix and list of feature names
    """
    all_hpos = set()
    for patient in patients:
        all_hpos.update(patient['expanded_hpos'])

    all_hpos = sorted(all_hpos, key=lambda x: ic.get(x, 0), reverse=True)[:max_features]

    features = np.zeros((len(patients), len(all_hpos)))
    for i, patient in enumerate(patients):
        for j, hpo in enumerate(all_hpos):
            if hpo in patient['expanded_hpos']:
                features[i, j] = ic.get(hpo, 0)
    return features, all_hpos

def create_text_features(patients: List[Dict[str, Any]], term_info: Dict[str, Dict[str, Any]], max_features: int) -> Tuple[np.ndarray, List[str]]:
    """
    Create text features from HPO terms

    Args:
        patients: List of patient data dictionaries
        term_info: Dictionary of term information
        max_features: Maximum number of features to include

    Returns:
        Feature matrix and list of feature names
    """
    def get_hpo_text(hpo_id: str) -> str:
        info = term_info.get(hpo_id, {})
        text = f"{info.get('name', '')} {info.get('definition', '')} {' '.join(info.get('synonyms', []))}"
        return text

    hpo_texts = [' '.join(get_hpo_text(hpo) for hpo in patient['expanded_hpos']) for patient in patients]
    
    stop_words = list(stopwords.words('english'))
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X_text = vectorizer.fit_transform(hpo_texts)
    
    return X_text.toarray(), vectorizer.get_feature_names_out()

def create_graph_embeddings(hpo_graph: nx.Graph, dimensions: int) -> Dict[str, np.ndarray]:
    """
    Create graph embeddings for HPO terms

    Args:
        hpo_graph: The HPO ontology graph
        dimensions: Number of dimensions for the embeddings

    Returns:
        Embeddings for each HPO term
    """
    def get_term_context(term: str) -> List[str]:
        context = [term]
        context.extend(hpo_graph.predecessors(term))
        context.extend(hpo_graph.successors(term))
        return context

    sentences = [get_term_context(node) for node in hpo_graph.nodes()]
    model = Word2Vec(sentences, vector_size=dimensions, window=5, min_count=1, workers=4)
    embeddings = {node: model.wv[node] for node in hpo_graph.nodes()}
    return embeddings

def create_embedding_features(patients: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create embedding features for patients

    Args:
        patients: List of patient data dictionaries
        embeddings: Embeddings for each HPO term

    Returns:
        Feature matrix
    """
    features = np.zeros((len(patients), len(next(iter(embeddings.values())))))
    for i, patient in enumerate(patients):
        patient_embedding = np.mean([embeddings.get(hpo, np.zeros_like(next(iter(embeddings.values())))) for hpo in patient['expanded_hpos']], axis=0)
        features[i] = patient_embedding
    return features

def load_config(config_file):
    """
    Load configuration from a file

    Args:
        config_file: Path to the configuration file

    Returns:
        Loaded configuration
    """
    config = configparser.ConfigParser()
    config.read(os.path.join(project_root, config_file))
    return config

def main(config_file):
    """
    Main function to process HPO data and create feature matrices

    Args:
        config_file: Path to the configuration file
    """
    config = load_config(config_file)

    hpo_file = os.path.join(project_root, config['Paths']['hpo_file'])
    hpoa_file = os.path.join(project_root, config['Paths']['hpoa_file'])
    patient_file = os.path.join(project_root, config['Paths']['patient_file'])
    output_dir = os.path.join(project_root, config['Paths']['output_dir'])

    max_weighted_hpo_features = config.getint('Features', 'max_weighted_hpo_features')
    max_text_features = config.getint('Features', 'max_text_features')
    embedding_dimensions = config.getint('Features', 'embedding_dimensions')

    target_disease = config['Processing']['target_disease']

    logger.info("Parsing HPO ontology...")
    hpo_graph, term_info = parse_hpo_ontology(hpo_file)
    
    logger.info("Processing HPOA annotations...")
    disease_to_hpo, hpo_to_disease = process_hpoa_annotations(hpoa_file)
    
    logger.info("Processing patient data...")
    patients = process_patient_data(patient_file)

    logger.info("Connecting data...")
    processed_patients = connect_data(patients, hpo_graph, disease_to_hpo, hpo_to_disease)

    logger.info("Calculating information content...")
    ic = calculate_ic(disease_to_hpo)

    logger.info("Creating weighted HPO features...")
    X_weighted_hpo, weighted_hpo_features = create_weighted_hpo_features(processed_patients, ic, max_features=max_weighted_hpo_features)

    logger.info("Creating text features...")
    X_text, text_features = create_text_features(processed_patients, term_info, max_features=max_text_features)

    logger.info("Creating graph embeddings...")
    hpo_embeddings = create_graph_embeddings(hpo_graph, dimensions=embedding_dimensions)

    logger.info("Creating embedding features...")
    X_embeddings = create_embedding_features(processed_patients, hpo_embeddings)

    logger.info("Combining features...")
    X_combined = np.hstack([X_weighted_hpo, X_text, X_embeddings])

    logger.info("Preparing target variable...")
    y = np.array([1 if patient.get('disease') == target_disease else 0 for patient in processed_patients])

    logger.info("Saving processed data...")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_feature.npy'), X_combined)
    np.save(os.path.join(output_dir, 'y_target.npy'), y)
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump({
            'weighted_hpo_features': weighted_hpo_features,
            'text_features': text_features.tolist(),
            'embedding_features': [f'embedding_{i}' for i in range(X_embeddings.shape[1])]
        }, f)

    logger.info(f"Feature matrix shape: {X_combined.shape}")
    logger.info(f"Target shape: {y.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process HPO data and create feature matrices")
    parser.add_argument("--config", default="config/config.ini", help="Path to the configuration file")
    args = parser.parse_args()

    main(args.config)