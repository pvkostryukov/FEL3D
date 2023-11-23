dim = 3
nodes = 5
h_nodes = nodes // 2
γ = 1  # / 1.043218

E_0 = 1.5

extensions = [['.1sh', '.1exsh', '.1', '.1ex', '.1el', '.1exl'], 
              ['.dat03', '.dat0329', '.dat', '.she', '.dat29', '.she29']]

m_unit = 939.5656
h_bar_c = 197.327
sqrt_pi = 1.7724538509055159
pi2 = 9.869604401089358

fact = 2 * m_unit / (pi2 * h_bar_c ** 2)

out_dict = {'Z': '', 'A_prime': '', 'A_rest': '', 'B_cf': '',
            'E*': '', 'e_n': ''}
n_emission = 6

E_Wig = 5
E_γ_thres = 5