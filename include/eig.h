#ifndef PRR_EIG_H
#define PRR_EIG_H

/**
 * @brief Calcule les valeurs et vecteurs propres d'une matrice, triés par ordre décroissant de magnitude des valeurs propres.
 *
 * @param N La dimension de la matrice
 * @param A Une matrice de taille NxN, stockée en column major. Cette matrice est modifiée après un appel à cette fonction.
 * @param lda Le nombre d'éléments entre 2 colonnes de `A`, doit être >= `N`.
 * @param limit Le nombre de valeurs/vecteurs propres à calculer.
 * @param out_eigvals En sortie, contient un pointeur vers un tableau contenant les parties réelles et imaginaires des valeurs propres triées.
 * @param out_eigvecs En sortie, contient un pointeur vers les vecteurs propres triés.
 *
 * @return Le nombre de valeurs/vecteurs propres calculées.
 */
int sorted_eigvals(int N, double* A, int lda, int limit, double** out_eigvals, double** out_eigvecs);

#endif // !PRR_EIG_H
