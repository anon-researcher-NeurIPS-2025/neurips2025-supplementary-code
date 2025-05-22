# hunger_index and equality_index metrics

def hunger_index(gamma, A, t_vector, players, individual=False):
    t = t_vector[-1]
    res = {}
    for index_player, player in enumerate(players):
        num = sum(gamma**(-1 * (t - t_t)) * A[index_player][index_t_t] for index_t_t, t_t in enumerate(t_vector))
        den = sum(gamma**(-1 * (t - t_t)) for t_t in t_vector)
        res[player] = num / den
    if individual:
        return res
    return sum(res[player] for player in players) / len(players)

def equality_index(A, t_vector, players):
    A = sorted(A, key=sum)
    players = [player + 1 for player in players]  # Make players start from 1
    N = len(players)
    T_a = sum(sum(A[i]) for i in range(N))
    den = T_a * sum(players)
    num = N * sum(
        sum(
            sum(A[j][t] for t, _ in enumerate(t_vector))
            for j in range(0, i + 1)
        ) for i in range(N)
    )
    return num / den if den > 0 else 1

def calculate_hunger(range_episode, n, consumed_apples):
    hunger = []
    gamma = 1.03
    t_vector = range(range_episode)
    players = range(n)
    for i_t in t_vector:
        t_vector_edited = t_vector[:i_t+1]
        A_edited = [consumed_apples[player][:i_t+1] for player in players]
        hunger.append(hunger_index(gamma, A_edited, t_vector_edited, players))
    return hunger

def calculate_equality(range_episode, n, consumed_apples):
    equality = []
    t_vector = range(range_episode)
    players = range(n)
    for i_t in t_vector:
        t_vector_edited = t_vector[:i_t+1]
        A_edited = [consumed_apples[player][:i_t+1] for player in players]
        equality.append(equality_index(A_edited, t_vector_edited, players))
    return equality
