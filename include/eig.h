#ifndef PRR_EIG_H
#define PRR_EIG_H

/**
 * @brief Calcule les valeurs et vecteurs propres d'une matrice, triés par ordre décroissant de magnitude des valeurs propres.
 *
 * @param N La dimension de la matrice
 * @param A Une matrice de taille NxN, stockée en column major. Cette matrice est modifiée après un appel à cette fonction.
 * @param lda Le nombre d'éléments entre 2 colonnes de `A`, doit être >= `N`.
 * @param eigvals_re Tableau de double de taille `N`. En sortie, contient les parties réelles des valeurs propres triées.
 * @param eigvals_im Tableau de double de taille `N`. En sortie, contient les parties imaginaires des valeurs propres triées.
 * @param out_eigvecs Tableau de doubles de taille `N*N`. En sortie, contient les vecteurs propres triés.
 *
 * @return int La valeur de retour de la fonction BLAS `dgeev`
 */
int sorted_eigvals(int N, double* A, int lda, double* eigvals_re, double* eigvals_im, double* eigvecs);

#endif // !PRR_EIG_H
