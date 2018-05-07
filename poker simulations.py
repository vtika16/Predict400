from __future__ import division
from collections import namedtuple, Counter
import doctest
import random
import time

CARDS_PER_HAND = 5
VALUES = 13
KINDS = 4
ACCURACY = 2 * 10 ** 6

Card = namedtuple("Card", ["value", "kind"])

def all_equal(lst):
    """
    >>> all_equal([1,1,1,1])
    True
    >>> all_equal([1,2,3])
    False
    """
    return len(set(lst)) == 1

def couples(lst):
    """
    >>> couples([1,5,7])
    [[1, 5], [5, 7]]
    """
    return [ [curr, lst[index + 1]]
                   for index, curr in enumerate(lst[:-1])]

def one_by_one_increasing(lst):
    """
    >>> one_by_one_increasing([5, 6, 7, 8])
    True
    """
    return all(nexxt == previous + 1
                   for previous, nexxt in couples(lst))

def most_common(lst):
    """
    >>> most_common([1,2,2,2,3])
    2
    """
    return Counter(lst).most_common()[0][0]

def most_common_count(lst):
    """
    >>> most_common_count([4,4,4,1,1])
    3
    >>> most_common_count([7,3,1,7,8])
    2
    """
    return lst.count(most_common(lst))

def first_true(arg, funcs):
    """
    >>> def false(n): return False
    >>> def even(n): return n % 2 == 0
    >>> first_true(14, [false, even]).__name__
    'even'
    """
    for f in funcs:
        if f(arg):
            return f

def values(cards):
    """
    >>> values([ Card(12, 3), Card(5, 2), Card(8, 3)])
    [12, 5, 8]
    """
    return [c.value for c in cards]

def kinds(cards):
    """
    >>> kinds([ Card(12, 2), Card(5, 3)])
    [2, 3]
    """
    return [c.kind for c in cards]

def is_straight_flush(hand):
    """
    >>> is_straight_flush(Card(x, 2) for x in range(4, 9))
    True
    """
    return is_flush(hand) and is_straight(hand)

def is_flush(hand):
    """
    >>> is_flush(Card(x, 2) for x in [3,6,1,10])
    True
    """
    return all_equal(kinds(hand))

def is_four_of_a_kind(hand):
    """
    >>> is_four_of_a_kind([ Card(5, 2), Card(5, 3), Card(5, 1), Card(5, 4), Card(8, 3) ])
    True
    """
    return most_common_count(values(hand)) == 4

def is_three_of_a_kind(hand):
    """
    >>> is_three_of_a_kind( [Card(12, 2)]*3 + [Card(2, 3), Card(9, 2)])
    True
    >>> is_three_of_a_kind( [Card(2, 1), Card(3, 2), Card(3, 1), Card(5, 1), Card(9, 4)] )
    False
    """
    return most_common_count(values(hand)) == 3

def is_pair(hand):
    """
    >>> is_pair( [Card(2, 1), Card(3, 2), Card(3, 1), Card(5, 1), Card(9, 4)] )
    True
    """
    return most_common_count(values(hand)) == 2

def is_straight(hand):
    """
    >>> is_straight( [Card(value, random.randint(0,4)) for value in range(0,5)] )
    True
    """
    return one_by_one_increasing(sorted(values(hand)))

def is_two_pair(hand):
    """
    >>> is_two_pair([Card(1, 1), Card(3, 2), Card(3, 1), Card(9, 1), Card(9, 4)])
    True
    >>> is_two_pair( [Card(2, 1), Card(3, 2), Card(3, 1), Card(5, 1), Card(9, 4)] )
    False
    """
    return is_pair(hand) and is_pair([c for c in hand if c.value != most_common(values(hand))])

def is_full_house(hand):
    """
    >>> is_full_house([Card(3, 1), Card(3, 2), Card(3, 1), Card(9, 1), Card(9, 4)])
    True
    """
    return is_three_of_a_kind(hand) and is_pair([c for c in hand if c.value != most_common(hand).value])

def is_nothing(hand):
    """
    A hand is always at least nothing
    >>> is_nothing(Card(random.randint(0,12), random.randint(0,3)) for _ in range(CARDS_PER_HAND))
    True
    """
    return True

def poker_value(hand, possible_scores=[is_straight_flush, is_four_of_a_kind, is_full_house,
                        is_flush, is_straight, is_three_of_a_kind, is_two_pair, is_pair, is_nothing]):
    """
    >>> poker_value([ Card(5, 1), Card(7, 2), Card(9, 3), Card(10, 1), Card(10, 1) ])
    'pair'
    >>> poker_value([ Card(val, 3) for val in range(2, 7)])
    'straight_flush'
    """
    return first_true(hand, possible_scores).__name__[3:]

def poker_deck(max_value=VALUES, number_of_kinds=KINDS):
    """
    >>> len(poker_deck())
    52
    """
    return [ Card(value, kind)
                 for value in range(max_value)
                     for kind in range(number_of_kinds)]

def poker_percentages(accuracy):
    deck = poker_deck()
    occurencies = Counter( poker_value(random.sample(deck, CARDS_PER_HAND))
                        for _ in range(accuracy))
    return list(sorted(((name, round((occurencies[name] / accuracy * 100), CARDS_PER_HAND) )
                           for name in occurencies), key=lambda x: x[1], reverse=True))

def main():
    start = time.time()
    print("Poker probabilities:")
    print("\n".join(map(str, poker_percentages(ACCURACY))))
    print("Time taken: {}".format(time.time() - start))

if __name__ == "__main__":
    doctest.testmod()
    main()
    
    
shufflings = lambda n: reduce(operator.mul, range(1, n+1))
permutations = lambda n, r: reduce(operator.mul, range(n-r+1, n+1))
combinations = lambda n, r: permutations(n, r) / shufflings(r)

----------------------------------------------------------------------------------------------------------------------------------
cards = 52
aces = 4
card_probability = 4/52
print(round(card_probability, 2))

card_prob_perc = card_probability * 100
print(str(round(card_prob_perc, 0)) + '%')

def event_prob(event_outcomes, sample_space):
    probability = (event_outcomes / sample_space) * 100
    return round(probability, 1)
    
# Sample Space
cards = 52

# Determine the probability of drawing a heart
hearts = 13
heart_probability = event_prob(hearts, cards)

# Determine the probability of drawing a face card
face_cards = 12
face_card_probability = event_prob(face_cards, cards)

# Determine the probability of drawing the queen of hearts
queen_of_hearts = 1
queen_of_hearts_probability = event_prob(queen_of_hearts, cards)

# Print each probability
print(str(heart_probability) + '%')
print(str(face_card_probability) + '%')
print(str(queen_of_hearts_probability) + '%')

# Sample Space
cards = 52
cards_drawn = 1 
cards = cards - cards_drawn 

# Determine the probability of drawing an Ace after drawing a King on the first draw
aces = 4
ace_probability1 = event_prob(aces, cards)

# Determine the probability of drawing an Ace after drawing an Ace on the first draw
aces_drawn = 1
aces = aces - aces_drawn
ace_probability2 = event_prob(aces, cards)

# Print each probability
print(ace_probability1)
print(ace_probability2)

-----------------------------------------------------------------------------------------------------

def card_prob(possible_outs, cardsleft):
    probability = (possible_outs / cardsleft) * 100
    return round(probability, 1)

#number of cards
cards = 52
hand = 2
community = 5
cards_left = cards - (hand + community)
print(str(cards_left) + ' cards left')


#needed outcome
clubs = 13
clubs_drawn = 2

outs = clubs - clubs_drawn
print(str(outs) + ' available')

#Determine flush probability
club_flush_prob = card_prob(outs,cards_left)
print(str(club_flush_prob) + '% chance of hitting a flush')

#needed outcome
aces_and_nines = 8
aces_and_nines_drawn = 2

outs = aces_and_nines - aces_and_nines_drawn
print(str(outs) + ' available')

#Determine flush probability
full_house_prob = card_prob(outs,cards_left)
print(str(full_house_prob) + '% chance of hitting a full house')



#number of cards
cards = 52
hand = 2
community = 3
cards_left = cards - (hand + community)
print(str(cards_left) + ' cards left')

#needed outcome
aces_and_nines = 8
aces_and_nines_drawn = 4

outs = aces_and_nines - aces_and_nines_drawn
print(str(outs) + ' available')

#Determine flush probability
full_house_prob = card_prob(outs,cards_left)
print(str(full_house_prob) + '% chance of hitting a full house')


#needed outcome
clubs = 13
clubs_drawn = 3

outs = clubs - clubs_drawn
print(str(outs) + ' available')

#Determine flush probability
club_flush_prob = card_prob(outs,cards_left)
print(str(club_flush_prob) + '% chance of hitting a flush')


#number of cards
cards = 52
community = 5
matts_cards = 2
cards_left = cards - (community + matts_cards)
print(str(cards_left) + ' cards left')

#needed outcome
aces = 4
aces_taken = 2

hand_card = aces - aces_taken
print(str(hand_card) + ' available')

#Determine flush probability
pocket_rockets_prob = card_prob(hand_card,cards_left)
print(str(pocket_rockets_prob) + '% chance of having 2 aces in his hand')




##Number of cards in deck and aces in deck
cards = 52
aces = 4

##odds of one player being dealt an ace
odds of getting one ace