#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de Federated Learning sur MNIST comparant :
1. Le modèle centralisé (baseline)
2. FedAvg avec tous les edges participants
3. FedAvg avec une fraction fixe (ex. 30%) d'edges participants
Les résultats (loss & accuracy) sont suivis et représentés graphiquement.
Les graphiques sont sauvegardés en format PNG et PDF pour intégration dans le rapport.

Auteur : [Votre Nom]
Date   : [Date]
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Import de vos modules locaux (assurez-vous que fl_dataquest.py et fl_model.py sont dans le PYTHONPATH)
import fl_dataquest  # pour get_data, get_dataset
import fl_model      # pour MyModel

# -------------------------------
# Fonctions utilitaires FedAvg
# -------------------------------
def make_edges(image_list, label_list, num_edges=10, verbose=0):
    """
    Répartit les données (images et labels) entre 'num_edges' clients de façon aléatoire.

    Arguments:
        image_list (list): liste des images (tableaux NumPy)
        label_list (list): liste des labels (one-hot encodés)
        num_edges (int): nombre de clients
        verbose (int): niveau de verbosité

    Retourne:
        dict: un dictionnaire où chaque clé est 'edge_i' et la valeur est un tf.data.Dataset
    """
    edge_names = [f"edge_{i}" for i in range(num_edges)]
    data = list(zip(image_list, label_list))
    random.shuffle(data)  # Mélanger les données

    size = len(data) // num_edges
    shards = [data[i:i + size] for i in range(0, size * num_edges, size)]
    edges = {edge_names[i]: shards[i] for i in range(num_edges)}

    edges_batched = {}
    for en, shard_data in edges.items():
        il, ll = zip(*shard_data)
        il = list(il)
        ll = list(ll)
        if verbose > 0:
            print(f"{en} => #images = {len(il)}; shapeX = {il[0].shape}")
        ds = tf.data.Dataset.from_tensor_slices((il, ll)).shuffle(len(il)).batch(32)
        edges_batched[en] = ds
    return edges_batched


def make_edges_non_iid(X, y, num_edges=5, verbose=0):
    """
    Crée un partitionnement Non-IID.
    Exemple pour 5 edges :
      edge_0 : chiffres [0,1]
      edge_1 : chiffres [2,3]
      edge_2 : chiffres [4,5]
      edge_3 : chiffres [6,7]
      edge_4 : chiffres [8,9]

    Arguments :
        X (list): liste d'images
        y (list): liste de labels (one-hot)
        num_edges (int): nombre d'edges (pour ce mapping, <= 5)
        verbose (int): niveau de verbosité

    Retourne :
        dict: chaque clé correspond à un edge et la valeur est un tf.data.Dataset
    """
    digits_per_edge = 2  # 2 chiffres par edge
    edges_map = {}
    for i in range(num_edges):
        start_digit = 2 * i
        local_digits = list(range(start_digit, start_digit + digits_per_edge))
        edges_map[f"edge_{i}"] = local_digits

    edges_dict = {f"edge_{i}": [] for i in range(num_edges)}
    for (img, label) in zip(X, y):
        class_idx = np.argmax(label)
        for en, digit_list in edges_map.items():
            if class_idx in digit_list:
                edges_dict[en].append((img, label))
                break

    edges_batched = {}
    for en, shard_data in edges_dict.items():
        random.shuffle(shard_data)
        il, ll = zip(*shard_data)
        il = list(il)
        ll = list(ll)
        ds = tf.data.Dataset.from_tensor_slices((il, ll)).shuffle(len(il)).batch(32)
        edges_batched[en] = ds
        if verbose > 0:
            print(f"{en} => #images = {len(il)} (digits = {edges_map[en]})")
    return edges_batched


def weight_scaling_factor(edges, edge_name):
    """
    Calcule la fraction (# d'exemples dans un edge) / (# total d'exemples).
    On suppose ici un batch de taille 32.
    """
    total_count = sum(len(edges[en]) for en in edges.keys()) * 32
    local_count = len(edges[edge_name]) * 32
    return local_count / total_count


def scale_model_weights(weights, scalar):
    """
    Multiplie chaque tenseur de poids par le scalaire 'scalar'.
    """
    return [w * scalar for w in weights]


def sum_scaled_weights(scaled_weight_list):
    """
    Agrège les poids mis à l'échelle (somme layer-wise).
    Cela correspond à l'opération FedAvg.
    """
    new_weight = []
    for layers_tuple in zip(*scaled_weight_list):
        layer_sum = tf.math.reduce_sum(layers_tuple, axis=0)
        new_weight.append(layer_sum)
    return new_weight

# -------------------------------
# Expérimentation et visualisation
# -------------------------------
if __name__ == '__main__':
    # Fixer les seed pour la reproductibilité
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    # ----------------------------------------------------
    # a) Chargement et préparation des données MNIST
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(
        img_path='./Mnist/trainingSet/trainingSet/',  # Adapter selon votre arborescence
        verbose=1
    )
    dtt, dts = fl_dataquest.get_dataset(X_train, X_test, y_train, y_test,
                                        batch_size=32, verbose=1)

    # ----------------------------------------------------
    # b) Entraînement centralisé (baseline)
    # ----------------------------------------------------
    cmodel = fl_model.MyModel(input_shape, nbclasses=10)
    cmodel.fit_it(trains=dtt, epochs=5, tests=dts, verbose=1)
    loss_c, acc_c = cmodel.evaluate(dts, verbose=1)
    print(f"\n[Modèle centralisé] => Loss = {loss_c:.4f}  Acc = {acc_c:.4f}")
    central_results = {"loss": loss_c, "accuracy": acc_c}

    # ----------------------------------------------------
    # c) Création des edges
    # ----------------------------------------------------
    # Pour cet exemple, nous utilisons un partitionnement classique pour comparer
    num_edges = 5
    edges_classic = make_edges(X_train, y_train, num_edges=num_edges, verbose=1)
    # Vous pouvez aussi utiliser le partitionnement Non-IID :
    edges_non_iid = make_edges_non_iid(X_train, y_train, num_edges=num_edges, verbose=1)

    # ----------------------------------------------------
    # d) Expérience FedAvg - cas 1 : Tous les edges participant
    # ----------------------------------------------------
    num_rounds = 5
    edge_epoch = 1

    fedmodel_all = fl_model.MyModel(input_shape, nbclasses=10)
    fedmodel_all.set_weights(cmodel.get_weights())

    rounds_all = []
    fedall_acc = []
    fedall_loss = []

    for r in range(num_rounds):
        print(f"\n=== FedAvg (TOUS les edges) - Round {r + 1}/{num_rounds} ===")
        scaled_local_weight_list = []
        for en in edges_classic.keys():
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedmodel_all.get_weights())
            dtrain_edge = edges_classic[en]
            local_model.fit_it(trains=dtrain_edge, epochs=edge_epoch, tests=None, verbose=0)
            sf = weight_scaling_factor(edges_classic, en)
            w_scaled = scale_model_weights(local_model.get_weights(), sf)
            scaled_local_weight_list.append(w_scaled)
            K.clear_session()
        fedavg_weights = sum_scaled_weights(scaled_local_weight_list)
        fedmodel_all.set_weights(fedavg_weights)
        loss_fed, acc_fed = fedmodel_all.evaluate(dts, verbose=0)
        print(f"    Round {r + 1}: Loss = {loss_fed:.4f}, Acc = {acc_fed:.4f}")
        rounds_all.append(r + 1)
        fedall_loss.append(loss_fed)
        fedall_acc.append(acc_fed)

    # ----------------------------------------------------
    # e) Expérience FedAvg - cas 2 : Fraction fixe d'edges (ex: 30% participants)
    # ----------------------------------------------------
    fraction = 0.3
    n_participate = max(1, int(num_edges * fraction))
    all_edge_names = list(edges_classic.keys())
    # Sélection unique et fixe des edges participants sur tout le processus
    participating_edges = random.sample(all_edge_names, k=n_participate)
    print(f"\nEdges participants fixes : {participating_edges}")

    fedmodel_frac = fl_model.MyModel(input_shape, nbclasses=10)
    fedmodel_frac.set_weights(cmodel.get_weights())

    rounds_frac = []
    fedfrac_acc = []
    fedfrac_loss = []

    for r in range(num_rounds):
        print(f"\n=== FedAvg (Fraction fixe - {fraction * 100:.0f}% des edges) - Round {r + 1}/{num_rounds} ===")
        scaled_local_weight_list = []
        for en in participating_edges:
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedmodel_frac.get_weights())
            dtrain = edges_classic[en]
            local_model.fit_it(trains=dtrain, epochs=edge_epoch, tests=None, verbose=0)
            sf = weight_scaling_factor(edges_classic, en)
            w_scaled = scale_model_weights(local_model.get_weights(), sf)
            scaled_local_weight_list.append(w_scaled)
            K.clear_session()
        fedavg_weights = sum_scaled_weights(scaled_local_weight_list)
        fedmodel_frac.set_weights(fedavg_weights)
        loss_fed, acc_fed = fedmodel_frac.evaluate(dts, verbose=0)
        print(f"    Round {r + 1}: Loss = {loss_fed:.4f}, Acc = {acc_fed:.4f}")
        rounds_frac.append(r + 1)
        fedfrac_loss.append(loss_fed)
        fedfrac_acc.append(acc_fed)

    # ----------------------------------------------------
    # f) Évaluation finale et comparaison
    # ----------------------------------------------------
    final_loss_all, final_acc_all = fedmodel_all.evaluate(dts, verbose=1)
    final_loss_frac, final_acc_frac = fedmodel_frac.evaluate(dts, verbose=1)
    print(f"\n[Modèle FedAvg - Tous les edges] => Loss = {final_loss_all:.4f}, Acc = {final_acc_all:.4f}")
    print(f"[Modèle FedAvg - Fraction fixe]    => Loss = {final_loss_frac:.4f}, Acc = {final_acc_frac:.4f}")
    print(f"\nComparaison:")
    print(f" - Centralisé       => Acc = {central_results['accuracy']:.4f}")
    print(f" - FedAvg (tous)    => Acc = {final_acc_all:.4f}")
    print(f" - FedAvg (fraction)=> Acc = {final_acc_frac:.4f}")

    # ----------------------------------------------------
    # g) Visualisation des résultats avec Matplotlib
    # ----------------------------------------------------
    # Création d'une figure avec deux sous-graphiques
    plt.figure(figsize=(14, 6))

    # Graphique de l'évolution de l'Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(rounds_all, fedall_acc, marker='o', linestyle='-', color='blue', label='FedAvg - Tous les edges')
    plt.plot(rounds_frac, fedfrac_acc, marker='o', linestyle='-', color='green', label='FedAvg - Fraction fixe')
    plt.hlines(central_results["accuracy"], rounds_all[0], rounds_all[-1],
               colors='red', linestyles='dashed', label='Modèle centralisé')
    plt.title("Évolution de l'Accuracy")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Graphique de l'évolution de la Loss
    plt.subplot(1, 2, 2)
    plt.plot(rounds_all, fedall_loss, marker='o', linestyle='-', color='blue', label='FedAvg - Tous les edges')
    plt.plot(rounds_frac, fedfrac_loss, marker='o', linestyle='-', color='green', label='FedAvg - Fraction fixe')
    plt.title("Évolution de la Loss")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Comparaison des performances de FedAvg")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Sauvegarde des graphiques en fichiers PNG et PDF
    plt.savefig("fedavg_comparison.png", format="png")
    plt.savefig("fedavg_comparison.pdf", format="pdf")
    plt.show()
