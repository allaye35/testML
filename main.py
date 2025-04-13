#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Comparaison de plusieurs méthodes de Federated Learning sur MNIST:
- Modèle centralisé (baseline)
- FedAvg (tous les clients)
- FedAvg (avec ~50% des clients hors-ligne)
- FedProx (avec terme proximal mu)
- SCAFFOLD (avec correcteurs locaux c_k)

Partitionnement Non-IID: (ex. 5 clients => edge_0: {0,1}, edge_1: {2,3}, ...)

Auteur   : [Votre Nom / Votre Équipe]
Date     : [Date]
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# == Imports locaux (à adapter) ==
import fl_dataquest  # pour get_data, get_dataset
import fl_model      # pour MyModel


# =========================================================================
# 1) Fonctions pour le partitionnement Non-IID et FedAvg
# =========================================================================

def make_edges_non_iid(X, y, num_edges=5, verbose=0):
    """
    Crée un partitionnement Non-IID de type :
       edge_0 : chiffres [0,1]
       edge_1 : chiffres [2,3]
       ...
    On suppose num_edges <= 5 pour couvrir 10 digits (2 par client).
    """
    digits_per_edge = 2
    edges_map = {}
    for i in range(num_edges):
        start_digit = 2 * i
        local_digits = list(range(start_digit, start_digit + digits_per_edge))
        edges_map[f"edge_{i}"] = local_digits

    edges_dict = {f"edge_{i}": [] for i in range(num_edges)}
    for (img, label) in zip(X, y):
        class_idx = np.argmax(label)  # 0..9
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
        ds = tf.data.Dataset.from_tensor_slices((il, ll))
        ds = ds.shuffle(len(il)).batch(32)
        edges_batched[en] = ds
        if verbose > 0:
            print(f"{en} => #images = {len(il)}; digits = {edges_map[en]}")
    return edges_batched


def weight_scaling_factor(edges, edge_name):
    """
    (# d'exemples dans edge_name) / (# total d'exemples),
    en supposant batch_size=32 pour tous.
    """
    total_count = sum(len(edges[en]) for en in edges.keys()) * 32
    local_count = len(edges[edge_name]) * 32
    return local_count / total_count


def scale_model_weights(weights, scalar):
    """ Multiplie chaque tenseur de poids par 'scalar'. """
    return [w * scalar for w in weights]


def sum_scaled_weights(scaled_weight_list):
    """
    Agrège les poids mis à l'échelle (somme).
    => correspond à FedAvg si la somme des scalars = 1
    """
    new_weight = []
    for layers_tuple in zip(*scaled_weight_list):
        layer_sum = tf.math.reduce_sum(layers_tuple, axis=0)
        new_weight.append(layer_sum)
    return new_weight


# =========================================================================
# 2) Implémentation FedProx
# =========================================================================

def local_training_fedprox(model, dataset, local_epochs=1, mu=0.1):
    """
    Entraînement local FedProx:
      - Ajout d'un terme proximal mu/2 * ||w_local - w_global||^2
    """
    global_weights = model.get_weights()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for _ in range(local_epochs):
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                logits = model.model(batch_x, training=True)
                loss = model.model.compiled_loss(batch_y, logits)
                # Ajout du terme proximal
                prox_term = 0
                for w_local, w_global in zip(model.get_weights(), global_weights):
                    prox_term += tf.reduce_sum(tf.square(w_local - w_global))
                loss += 0.5 * mu * prox_term

            grads = tape.gradient(loss, model.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.model.trainable_weights))

    return model.get_weights()


# =========================================================================
# 3) Implémentation SCAFFOLD (corrigé pour la soustraction couche/couche)
# =========================================================================

def local_training_scaffold(model, dataset, local_epochs, c_global, c_local):
    """
    SCAFFOLD:
      - c_global : liste de tenseurs pour correction globale
      - c_local  : liste de tenseurs pour correction locale
    """
    w_before = model.get_weights()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    for _ in range(local_epochs):
        for batch_x, batch_y in dataset:
            with tf.GradientTape() as tape:
                preds = model.model(batch_x, training=True)
                loss = model.model.compiled_loss(batch_y, preds)
            grads = tape.gradient(loss, model.model.trainable_weights)

            # Correction du gradient: g_corr = g + (c_global[i] - c_local[i])
            new_grads = []
            for g, w_l, c_loc_i, c_glo_i in zip(
                grads, model.model.trainable_weights, c_local, c_global
            ):
                c_diff = tf.convert_to_tensor(c_glo_i - c_loc_i, dtype=g.dtype)
                g_corr = g + c_diff
                new_grads.append(g_corr)

            optimizer.apply_gradients(zip(new_grads, model.model.trainable_weights))

    w_after = model.get_weights()

    # delta_w = w_after - w_before
    delta_w = [w_a - w_b for (w_a, w_b) in zip(w_after, w_before)]

    # Mise à jour c_local:
    #   c_local_new[i] = c_local[i] - c_global[i] + (1/(L*eta)) * (w_before[i] - w_after[i])
    eta = 0.01
    c_local_new = []
    for (c_l_i, c_g_i, w_b_i, w_a_i) in zip(c_local, c_global, w_before, w_after):
        c_ln = c_l_i - c_g_i + (1.0 / (local_epochs * eta)) * (w_b_i - w_a_i)
        c_local_new.append(c_ln)

    # delta_c = c_local_new - c_local
    delta_c = [cln - cl for (cln, cl) in zip(c_local_new, c_local)]
    return delta_w, delta_c, c_local_new


# =========================================================================
# 4) MAIN
# =========================================================================

if __name__ == '__main__':
    # Pour la reproductibilité
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # --------------------------------------------
    # a) Chargement MNIST et partitionnement Non-IID
    # --------------------------------------------
    X_train, X_test, y_train, y_test, input_shape = fl_dataquest.get_data(
        img_path='./Mnist/trainingSet/trainingSet/',
        verbose=1
    )
    dtrain_tf, dtest_tf = fl_dataquest.get_dataset(
        X_train, X_test, y_train, y_test,
        batch_size=32, verbose=1
    )

    num_edges = 5
    edges_non_iid = make_edges_non_iid(X_train, y_train, num_edges=num_edges, verbose=1)
    all_edge_names = list(edges_non_iid.keys())  # ex. ["edge_0", ..., "edge_4"]

    # --------------------------------------------
    # b) Modèle centralisé (baseline)
    # --------------------------------------------
    cmodel = fl_model.MyModel(input_shape, nbclasses=10)
    cmodel.fit_it(trains=dtrain_tf, epochs=5, tests=dtest_tf, verbose=1)
    loss_c, acc_c = cmodel.evaluate(dtest_tf, verbose=1)
    print(f"[Centralisé] => Loss={loss_c:.4f}, Acc={acc_c:.4f}")

    # Paramètres communs
    rounds = 5
    local_epochs = 1

    # =========================================================================
    # c) FedAvg (TOUS les edges)
    # =========================================================================
    fedavg_all = fl_model.MyModel(input_shape, nbclasses=10)
    fedavg_all.set_weights(cmodel.get_weights())

    hist_fedavg_all_loss = []
    hist_fedavg_all_acc = []

    for rd in range(rounds):
        print(f"\n[Round {rd+1}/{rounds}] FedAvg (ALL edges)")
        scaled_local_weight_list = []
        for en in edges_non_iid.keys():
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedavg_all.get_weights())
            local_model.fit_it(trains=edges_non_iid[en],
                               epochs=local_epochs,
                               tests=None, verbose=0)
            sf = weight_scaling_factor(edges_non_iid, en)
            w_scaled = scale_model_weights(local_model.get_weights(), sf)
            scaled_local_weight_list.append(w_scaled)
            K.clear_session()

        new_weights = sum_scaled_weights(scaled_local_weight_list)
        fedavg_all.set_weights(new_weights)
        l, a = fedavg_all.evaluate(dtest_tf, verbose=0)
        hist_fedavg_all_loss.append(l)
        hist_fedavg_all_acc.append(a)
        print(f" => FedAvg ALL => Loss={l:.4f}, Acc={a:.4f}")

    # =========================================================================
    # d) FedAvg (50% edges hors-ligne)
    # =========================================================================
    fedavg_part = fl_model.MyModel(input_shape, nbclasses=10)
    fedavg_part.set_weights(cmodel.get_weights())

    hist_fedavg_part_loss = []
    hist_fedavg_part_acc = []

    fraction_offline = 0.5
    n_offline = int(num_edges * fraction_offline)

    for rd in range(rounds):
        print(f"\n[Round {rd+1}/{rounds}] FedAvg (PARTIAL: 50% offline)")
        n_online = num_edges - n_offline
        participating_edges = random.sample(all_edge_names, k=n_online)

        scaled_local_weight_list = []
        for en in participating_edges:
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedavg_part.get_weights())
            local_model.fit_it(trains=edges_non_iid[en],
                               epochs=local_epochs,
                               tests=None, verbose=0)
            sf = weight_scaling_factor(edges_non_iid, en)
            w_scaled = scale_model_weights(local_model.get_weights(), sf)
            scaled_local_weight_list.append(w_scaled)
            K.clear_session()

        # Agrégation
        if len(scaled_local_weight_list) > 0:
            new_weights = sum_scaled_weights(scaled_local_weight_list)
            fedavg_part.set_weights(new_weights)

        # Évaluation
        l, a = fedavg_part.evaluate(dtest_tf, verbose=0)
        hist_fedavg_part_loss.append(l)
        hist_fedavg_part_acc.append(a)
        print(f" => FedAvg PART => Loss={l:.4f}, Acc={a:.4f}")

    # =========================================================================
    # e) FedProx
    # =========================================================================
    mu = 0.1
    fedprox = fl_model.MyModel(input_shape, nbclasses=10)
    fedprox.set_weights(cmodel.get_weights())

    hist_fedprox_loss = []
    hist_fedprox_acc = []

    for rd in range(rounds):
        print(f"\n[Round {rd+1}/{rounds}] FedProx mu={mu}")
        scaled_local_weight_list = []
        global_w = fedprox.get_weights()

        for en in edges_non_iid.keys():
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(global_w)
            w_local = local_training_fedprox(local_model,
                                             edges_non_iid[en],
                                             local_epochs=local_epochs,
                                             mu=mu)
            sf = weight_scaling_factor(edges_non_iid, en)
            w_scaled = scale_model_weights(w_local, sf)
            scaled_local_weight_list.append(w_scaled)
            K.clear_session()

        new_w = sum_scaled_weights(scaled_local_weight_list)
        fedprox.set_weights(new_w)
        l, a = fedprox.evaluate(dtest_tf, verbose=0)
        hist_fedprox_loss.append(l)
        hist_fedprox_acc.append(a)
        print(f" => FedProx => Loss={l:.4f}, Acc={a:.4f}")

    # =========================================================================
    # f) SCAFFOLD (avec correcteurs c_global, c_local)
    # =========================================================================

    def zero_like(weights_list):
        return [np.zeros_like(w) for w in weights_list]

    fedscaffold = fl_model.MyModel(input_shape, nbclasses=10)
    fedscaffold.set_weights(cmodel.get_weights())

    c_global = zero_like(fedscaffold.get_weights())
    c_local_dict = {}
    for en in edges_non_iid.keys():
        c_local_dict[en] = zero_like(fedscaffold.get_weights())

    hist_scaffold_loss = []
    hist_scaffold_acc = []

    for rd in range(rounds):
        print(f"\n[Round {rd+1}/{rounds}] SCAFFOLD")
        delta_w_all = []
        delta_c_all = []

        for en in edges_non_iid.keys():
            local_model = fl_model.MyModel(input_shape, nbclasses=10)
            local_model.set_weights(fedscaffold.get_weights())
            ds_edge = edges_non_iid[en]

            dw, dc, c_loc_new = local_training_scaffold(
                local_model, ds_edge, local_epochs,
                c_global, c_local_dict[en]
            )
            c_local_dict[en] = c_loc_new
            sf = weight_scaling_factor(edges_non_iid, en)

            dw_scaled = [x * sf for x in dw]
            dc_scaled = [x * sf for x in dc]
            delta_w_all.append(dw_scaled)
            delta_c_all.append(dc_scaled)

            K.clear_session()

        agg_dw = sum_scaled_weights(delta_w_all)
        agg_dc = sum_scaled_weights(delta_c_all)

        # w(t+1) = w(t) + agg_dw
        current_w = fedscaffold.get_weights()
        new_w = [w + dw for (w, dw) in zip(current_w, agg_dw)]
        fedscaffold.set_weights(new_w)

        # c_global(t+1) = c_global + agg_dc (sum(sf)=1)
        c_global = [c_g + d_c for (c_g, d_c) in zip(c_global, agg_dc)]

        l, a = fedscaffold.evaluate(dtest_tf, verbose=0)
        hist_scaffold_loss.append(l)
        hist_scaffold_acc.append(a)
        print(f" => SCAFFOLD => Loss={l:.4f}, Acc={a:.4f}")

    # =========================================================================
    # g) Comparaison finale & Visualisation
    # =========================================================================

    # Récupération des performances finales
    final_favg_all_loss, final_favg_all_acc = hist_fedavg_all_loss[-1], hist_fedavg_all_acc[-1]
    final_favg_part_loss, final_favg_part_acc = hist_fedavg_part_loss[-1], hist_fedavg_part_acc[-1]
    final_fprox_loss, final_fprox_acc = hist_fedprox_loss[-1], hist_fedprox_acc[-1]
    final_fscaf_loss, final_fscaf_acc = hist_scaffold_loss[-1], hist_scaffold_acc[-1]

    print("\n===== RÉSUMÉ FINAL =====")
    print(f"Modèle Centralisé     : Loss={loss_c:.4f}, Acc={acc_c:.4f}")
    print(f"FedAvg (tous)         : Loss={final_favg_all_loss:.4f}, Acc={final_favg_all_acc:.4f}")
    print(f"FedAvg (50% offline)  : Loss={final_favg_part_loss:.4f}, Acc={final_favg_part_acc:.4f}")
    print(f"FedProx (mu={mu})     : Loss={final_fprox_loss:.4f}, Acc={final_fprox_acc:.4f}")
    print(f"SCAFFOLD              : Loss={final_fscaf_loss:.4f}, Acc={final_fscaf_acc:.4f}")

    rounds_range = list(range(1, rounds+1))

    # Figure pour comparer Accuracy et Loss
    plt.figure(figsize=(12,5))

    # -- Accuracy
    plt.subplot(1,2,1)
    plt.plot(rounds_range, hist_fedavg_all_acc,  marker='o', label="FedAvg (all)")
    plt.plot(rounds_range, hist_fedavg_part_acc, marker='s', label="FedAvg (partial)")
    plt.plot(rounds_range, hist_fedprox_acc,     marker='x', label="FedProx")
    plt.plot(rounds_range, hist_scaffold_acc,    marker='^', label="SCAFFOLD")
    plt.hlines(acc_c, 1, rounds, colors='r', linestyles='dashed', label='Centralisé')
    plt.title("Évolution de l'Accuracy (par round)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # -- Loss
    plt.subplot(1,2,2)
    plt.plot(rounds_range, hist_fedavg_all_loss,  marker='o', label="FedAvg (all)")
    plt.plot(rounds_range, hist_fedavg_part_loss, marker='s', label="FedAvg (partial)")
    plt.plot(rounds_range, hist_fedprox_loss,     marker='x', label="FedProx")
    plt.plot(rounds_range, hist_scaffold_loss,    marker='^', label="SCAFFOLD")
    plt.hlines(loss_c, 1, rounds, colors='r', linestyles='dashed', label='Centralisé')
    plt.title("Évolution de la Loss (par round)")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Comparaison FL: Central vs FedAvg (all/partial) vs FedProx vs SCAFFOLD")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Sauvegarde des graphes
    plt.savefig("comparison_fedavg_fedprox_scaffold_partial.png")
    plt.show()
