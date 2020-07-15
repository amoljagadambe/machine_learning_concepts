from ml_algorithms.viterbi import ExampleSet
import numpy


def viterbi_algorithm(A, states, emissions, prior_probabilities, B, sequence):
    probabilities = []

    # convert given sequence of strings to indices
    emissions_dict = dict(zip(emissions, list(range(len(emissions)))))
    emissions_sequence = []
    for item in sequence:
        emissions_sequence.append(emissions_dict[item])

    # first maximal probability
    probabilities.append(
        tuple(
            prior_probabilities[state] * B[state, emissions_sequence[0]]
            for state in range(len(states)))
    )

    for i in range(1, len(emissions_sequence)):
        previous_probabilities = probabilities[-1]
        current_probabilities = []
        for col in range(len(A[0, :])):
            p = max(
                previous_probabilities[state] * A[state, col] * B[col, emissions_sequence[i]]
                for state in range(len(states))
            )
            current_probabilities.append(p)
        probabilities.append(tuple(current_probabilities))

    # find the sequence of hidden states
    hidden_states_sequence = []
    for i in probabilities:
        hidden_state = states[numpy.argmax(i)]
        hidden_states_sequence.append(hidden_state)

    print(sequence)
    print(hidden_states_sequence)
    return probabilities, hidden_states_sequence


if __name__ == "__main__":
    transition_probability_matrix, hidden_states, observations, prior_probabilities, emission_probabilities_matrix, \
    emissions_sequence = ExampleSet.happy_sad_robot()

    P = viterbi_algorithm(
        transition_probability_matrix,
        hidden_states,
        observations,
        prior_probabilities,
        emission_probabilities_matrix,
        emissions_sequence)
