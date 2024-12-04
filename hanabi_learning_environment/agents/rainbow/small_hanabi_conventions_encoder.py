#   Orignal author: F Bredell
#   Last modified: 25 November 2024
#
#   This code serves as the conventions extraction and translation layers for the agent network. 
#   All conventions are stated before the init function and each class contains their own env processing layer 
#   to make the observation usable with the conventions. 

import numpy as np
from hanabi_learning_environment import rl_env

def format_legal_moves(legal_moves, action_dim):    # Extracted from run_experiment.py for ease of use and debugging
  """Returns formatted legal moves.

  This function takes a list of actions and converts it into a fixed size vector
  of size action_dim. If an action is legal, its position is set to 0 and -Inf
  otherwise.
  Ex: legal_moves = [0, 1, 3], action_dim = 5
      returns [0, 0, -Inf, 0, -Inf]

  Args:
    legal_moves: list of legal actions.
    action_dim: int, number of actions.

  Returns:
    a vector of size action_dim.
  """
  new_legal_moves = np.full(action_dim, -float('inf'))
  if legal_moves:
    new_legal_moves[legal_moves] = 0
  return new_legal_moves

class standalone_encoder():     
    #   Initial exploration of conventions as options specific to Small Hanabi. These conventions replace the full action space, i.e.
    #   does not contain action augmentation

    #-----------------List of conventions and their action ID-------------------
    #   0 - discard oldest card, unless colour hinted. if both colour hinted, discard oldest
    #   1 - hint next playable card (number)    
    #   2 - play value card playable
    #   3 - if score < 5, hint 5 to discard
    #   4 - if 5 is hinted and score < 5, discard 5
    #   5 - colour hint to keep card matching in discard pile and is not playable now
    #   6 - hint value card to indicate discard
    #   7 - discard value hinted card clearly not playable
    #   8 - play newest card if both are hinted/ambigious
    #   9 - play oldest card if information is implied
    #   10 - hint common information between 2 cards (colour or number)     # this could be the reason for problems
    #   11 - if oldest card value hinted and known as keep, discard second card
    #   12 - if implied knowledge + hinted knowledge gives card, play card
    #------------------------------------------------------------------------------
    def __init__(self, env):
        self.environment_action_space = env.action_space
        self.convention_action_space = 13

    def encode_action(self, current_player, other_player, agent_action, env): 
        # This function is the output layer after an agent chooses a convention and before the action is sent to the environment. 
        # It translates the chosen action/convention into an environment action based on the convention principles. 

        if agent_action == 0: 
            if int(self.current_player_hinted[0]/10) == 0: environment_action = 2
            elif int(self.current_player_hinted[0]/10) == 0 and int(self.current_player_hinted[1]/10) == 0: environment_action = 2
            elif int(self.current_player_hinted[0]/10) > 0 and int(self.current_player_hinted[1]/10) > 0: environment_action = 2
            else: environment_action = 3

        elif agent_action == 1:
            #extract current fireworks 
            current_firework_values = []
            for stack in env.stacks: current_firework_values.append(env.stacks[stack])
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    environment_action = 6 + (card%10)-1    #offsets action

        elif agent_action == 2:
            #extract current fireworks 
            current_firework_values = []
            for stack in env.stacks: current_firework_values.append(env.stacks[stack])
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10)-1 in current_firework_values:  #removes colour and decriments to match firework
                    environment_action = card_pos

                card_pos += 1

        elif agent_action == 3:
            if self.score < 5:
                environment_action = 10

        elif agent_action == 4:
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10) == 5: 
                    environment_action = card_pos + 2

                card_pos += 1

        elif agent_action == 5:
            for card in self.other_player_hand:
                if card in self.discard_pile: environment_action = int(card/10) + 3 #colour hint to keep

        elif agent_action == 6:
            #extract current fireworks 
            current_firework_values = []
            for stack in env.stacks: current_firework_values.append(env.stacks[stack])
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if card%10 <= current_firework_values[int(card/10)-1]:  #removes colour and decriments to match firework
                    environment_action = 5 + (card%10)    #offsets action

        elif agent_action == 7:
            #extract current fireworks 
            current_firework_values = []
            for stack in env.stacks: current_firework_values.append(env.stacks[stack])
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1]) and card%10 != 0:  #removes colour and decriments to match firework
                    environment_action = card_pos + 2

                card_pos += 1

        elif agent_action == 8: 
            environment_action = 1

        elif agent_action == 9: 
            environment_action = 0

        elif agent_action == 10:
            other_player_card_1 = self.other_player_hand[0]
            other_player_card_2 = self.other_player_hand[1]

            if other_player_card_1%10 == other_player_card_2%10: environment_action = 5 + other_player_card_1%10
            elif int(other_player_card_1/10) == int(other_player_card_2/10): environment_action = 3 + int(other_player_card_1/10)

        elif agent_action == 11: 
            environment_action = 3

        elif agent_action == 12:
            if env.players[current_player].implied_knowledge[0].count(0) > 0:
                environment_action = 0
            elif env.players[current_player].implied_knowledge[1].count(0) > 0:
                environment_action = 1
            
        return environment_action
    
    def available_conventions(self, current_player, other_player, env):
        # This function is at the input of the network to extract the current available conventions based on the current observation of a player. 
        # It also incorporates action masking and formatting to allow for the Dopamine agent to interpret it correctly.

        legal_conventions = []      #positional arguments correlate to actions in description

        # extract current fireworks, used in a few conventions
        current_firework_values = []
        for stack in env.stacks: current_firework_values.append(env.stacks[stack])

        #0
        legal_conventions.append(1) # agents must always have the option to discard
        # if self.hint_tokens < 3: legal_conventions.append(1)
        # else: legal_conventions.append(0)

        #1
        #check which card to hint in other player hand following firework
        if self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]: checker = True

            legal_conventions.append(int(checker))
        else: legal_conventions.append(0)

        #2
        checker = False
        card_pos = 0
        for card in self.current_player_hinted:
            if (card%10)-1 in current_firework_values: checker = True
            card_pos += 1

        legal_conventions.append(int(checker))

        #3
        if self.score < 5 and self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if (card%10) == 5: checker = True

            legal_conventions.append(int(checker))
        else: legal_conventions.append(0)

        #4
        if self.score < 5:
            checker = False
            for card in self.current_player_hinted:
                if (card%10) == 5: checker = True

            legal_conventions.append(int(checker))
        else: legal_conventions.append(0)

        #5
        if self.hint_tokens > 0:
            checker = False
            card_pos = 0
            for card in self.other_player_hand:
                if card in self.discard_pile and int(self.other_player_hinted[card_pos]/10) != 0: checker = True #colour hint to keep
                card_pos += 1
            legal_conventions.append(int(checker))
        else: legal_conventions.append(0)

        #6
        if self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if card%10 <= current_firework_values[int(card/10)-1]: checker = True

            legal_conventions.append(int(checker))
        else: legal_conventions.append(0)

        #7
        checker = False
        card_pos = 0
        for card in self.current_player_hinted:
            if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1]) and card%10 != 0: checker = True
            card_pos += 1

        legal_conventions.append(int(checker))

        #8
        # legal_conventions.append(0) #this could be problamatic, keep an eye out
        if (self.current_player_hinted[0] == self.current_player_hinted[1]) and self.current_player_hinted[0] != 0: legal_conventions.append(1)
        else: legal_conventions.append(0)
        
        #9
        # legal_conventions.append(1) #also could be problamatic
        if env.players[current_player].implied_knowledge[0].count(0) > 0: legal_conventions.append(1)  #this and convention 12 might be ambigious
        else: legal_conventions.append(0)

        #10
        # legal_conventions.append(0)
        if self.hint_tokens > 0:
            other_player_card_1 = self.other_player_hand[0]
            other_player_card_2 = self.other_player_hand[1]
            if other_player_card_1%10 == other_player_card_2%10 or int(other_player_card_1/10) == int(other_player_card_2/10): legal_conventions.append (1)
            else: legal_conventions.append(0)
        else: legal_conventions.append(0)

        #11
        if self.current_player_hinted[0] != 0: legal_conventions.append(1)
        else: legal_conventions.append(0)

        #12
        if env.players[current_player].implied_knowledge[0].count(0) > 0 or env.players[current_player].implied_knowledge[1].count(0) > 0: legal_conventions.append(1)  #this and convention 12 might be ambigious
        else: legal_conventions.append(0)

        return legal_conventions
    
    def make_env_usable(self, env):
        # This function extracts all the needed information from the current state and player observation to determine which conventions are available.
        # It receives the entire env as input to extract all the needed game features, but never gives an agent more information than it already
        # has access too. 

        current_player = env.state.cur_player()
        for player in range(env.state.num_players()):
            if player != current_player: other_player = player
        self.current_player = current_player
        self.other_player = other_player

        self.hint_tokens = env.state.information_tokens()

        fireworks_raw = env.state.fireworks()
        self.fireworks = fireworks_raw

        self.score = env.state.score()

        raw_hands = env.state.player_hands()

        current_player_hand = raw_hands[current_player]
        self.current_player_hand = current_player_hand
        card_counter = 0
        for card in current_player_hand:
            card_col = card._color
            card_val = card._rank
            self.current_player_hand[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        other_player_hand = raw_hands[other_player]
        self.other_player_hand = other_player_hand
        card_counter = 0
        for card in other_player_hand:
            card_col = card._color
            card_val = card._rank
            self.other_player_hand[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        raw_players_obs = env._make_observation_all_players()
        player_obs = raw_players_obs['player_observations'][current_player]['vectorized']
        legal_moves = raw_players_obs['player_observations'][current_player]['legal_moves_as_int']
        # self.legal_moves = format_legal_moves(legal_moves, env.num_moves())
        self.env_legal_moves = legal_moves

        raw_current_player_hinted = raw_players_obs['player_observations'][0]['card_knowledge'][current_player]
        raw_other_player_hinted = raw_players_obs['player_observations'][0]['card_knowledge'][other_player]

        self.current_player_hinted = []
        for hinted_card in raw_current_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.current_player_hinted.append(known)

        self.other_player_hinted = []
        for hinted_card in raw_other_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.other_player_hinted.append(known)

        raw_discard_pile = env.state.discard_pile()
        self.discard_pile = raw_discard_pile
        card_counter = 0
        for card in raw_discard_pile:
            card_col = card._color
            card_val = card._rank
            self.discard_pile[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        # pass #for debugging
    
class simple_combined_encoder():
    #   Action augmentation in Small Hanabi, with full convention list as well as a simplified list, that
    #   later became known as condensed list. The conventions are appened to the primitive action space.
    #   Currently the code is set up for the condensed version, but can be switched to full version by 
    #   uncommenting the needed parts.

    #-----------------List of conventions and their action ID-------------------
    #   0 (11) - discard oldest card, unless colour hinted. if both colour hinted, discard oldest
    #   1 (12)- hint next playable card (number)    
    #   2 (13)- play value card playable
    #   3 (14)- if score < 5, hint 5 to discard                     
    #   4 (15)- if 5 is hinted and score < 5, discard 5         #simplified stops here
    #   5 (16)- colour hint to keep card matching in discard pile and is not playable now
    #   6 (17)- hint value card to indicate discard
    #   7 (18)- discard value hinted card clearly not playable
    #   8 (19) - play newest card if both are hinted/ambigious
    #   9 (20) - play oldest card if information is implied
    #   10 (21) - hint common information between 2 cards (colour or number)     # this could be the reason for problems
    #   11 (22) - if oldest card value hinted and known as keep, discard second card
    #   12 (23) - if implied knowledge + hinted knowledge gives card, play card
    #------------------------------------------------------------------------------

    def __init__(self, env):
        self.environment_action_space = env.num_moves()
        # self.convention_action_space = self.environment_action_space + 13 # Full
        self.convention_action_space = self.environment_action_space + 5  # Simplified/augmented

    def encode_action(self, agent_action, env):  
        # This function is the output layer after an agent chooses a convention and before the action is sent to the environment. 
        # It translates the chosen action/convention into an environment action based on the convention principles.

        if agent_action < self.environment_action_space:
            return agent_action
        
        else: agent_action -= self.environment_action_space

        self.make_env_usable(env)
        
        if agent_action == 0: 
            if int(self.current_player_hinted[0]/10) == 0: environment_action = 0
            elif int(self.current_player_hinted[0]/10) == 0 and int(self.current_player_hinted[1]/10) == 0: environment_action = 0
            elif int(self.current_player_hinted[0]/10) > 0 and int(self.current_player_hinted[1]/10) > 0: environment_action = 0
            else: environment_action = 1

        elif agent_action == 1:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    environment_action = 6 + (card%10)-1    #offsets action

        elif agent_action == 2:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10)-1 in current_firework_values:  #removes colour and decriments to match firework
                    environment_action = card_pos + 2

                card_pos += 1

        elif agent_action == 3:
            if self.score < 5:
                environment_action = 10

        elif agent_action == 4:
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10) == 5: 
                    environment_action = card_pos

                card_pos += 1

        #--------------------Full---------------------------

        # elif agent_action == 5:
        #     for card in self.other_player_hand:
        #         if card in self.discard_pile: environment_action = int(card/10) + 3 #colour hint to keep

        # elif agent_action == 6:
        #     #extract current fireworks 
        #     current_firework_values = self.fireworks
        #     #check which card to hint in other player hand following firework
        #     for card in self.other_player_hand:
        #         if card%10 <= current_firework_values[int(card/10)-1]:  #removes colour and decriments to match firework
        #             environment_action = 5 + (card%10)    #offsets action

        # elif agent_action == 7:
        #     #extract current fireworks 
        #     current_firework_values = self.fireworks
        #     #check which card to hint in other player hand following firework
        #     card_pos = 0
        #     for card in self.current_player_hinted:
        #         if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1]) and card%10 != 0:  #removes colour and decriments to match firework
        #             environment_action = card_pos

        #         card_pos += 1

        # elif agent_action == 8: 
        #     environment_action = 1

        # elif agent_action == 9: 
        #     environment_action = 0

        # elif agent_action == 10:
        #     other_player_card_1 = self.other_player_hand[0]
        #     other_player_card_2 = self.other_player_hand[1]

        #     if other_player_card_1%10 == other_player_card_2%10: environment_action = 5 + other_player_card_1%10
        #     elif int(other_player_card_1/10) == int(other_player_card_2/10): environment_action = 3 + int(other_player_card_1/10)

        # elif agent_action == 11: 
        #     environment_action = 3

        # elif agent_action == 12:
        #     if env.players[current_player].implied_knowledge[0].count(0) > 0:
        #         environment_action = 0
        #     elif env.players[current_player].implied_knowledge[1].count(0) > 0:
        #         environment_action = 1
            
        return environment_action
    
    def available_conventions(self, env):
        # This function is at the input of the network to extract the current available conventions based on the current observation of a player. 
        # It also incorporates action masking and formatting to allow for the Dopamine agent to interpret it correctly.

        legal_conventions = []      #positional arguments correlate to actions in description

        self.make_env_usable(env)

        #environment mask
        legal_conventions = legal_conventions + self.env_legal_moves

        current_firework_values = self.fireworks

        #0
        # legal_conventions.append(1)
        if self.hint_tokens < 3: legal_conventions.append(0+self.environment_action_space)  #since agents now have access to primitive actions we can restrict this a bit more
        # else: legal_conventions.append(0)

        #1
        #check which card to hint in other player hand following firework
        if self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]: checker = True

            if checker: legal_conventions.append(1+self.environment_action_space)

        #2
        checker = False
        card_pos = 0
        for card in self.current_player_hinted:
            if (card%10)-1 in current_firework_values: checker = True
            card_pos += 1

        if checker: legal_conventions.append(2+self.environment_action_space)

        #3
        if self.score < 5 and self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if (card%10) == 5: checker = True

            if checker: legal_conventions.append(3+self.environment_action_space)

        #4
        if self.score < 5 and self.hint_tokens < 3:
            checker = False
            for card in self.current_player_hinted:
                if (card%10) == 5: checker = True

            if checker: legal_conventions.append(4+self.environment_action_space)

        #-------------------------------------------Full-------------------------------
        # #5
        # if self.hint_tokens > 0:
        #     checker = False
        #     card_pos = 0
        #     for card in self.other_player_hand:
        #         if card in self.discard_pile and int(self.other_player_hinted[card_pos]/10) != 0: checker = True #colour hint to keep
        #         card_pos += 1
        #     if checker: legal_conventions.append(5+self.environment_action_space)

        # #6
        # if self.hint_tokens > 0:
        #     checker = False
        #     for card in self.other_player_hand:
        #         if card%10 <= current_firework_values[int(card/10)-1]: checker = True

        #     if checker: legal_conventions.append(6+self.environment_action_space)

        # #7
        # if self.hint_tokens < 3:
        #     checker = False
        #     card_pos = 0
        #     for card in self.current_player_hinted:
        #         if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1]) and card%10 != 0: checker = True
        #         card_pos += 1

        #     if checker: legal_conventions.append(7+self.environment_action_space)

        # #8
        # # legal_conventions.append(0) #this could be problamatic, keep an eye out
        # if (self.current_player_hinted[0] == self.current_player_hinted[1]) and self.current_player_hinted[0] != 0: legal_conventions.append(8+self.environment_action_space)
        
        # #9
        # # legal_conventions.append(1) #also could be problamatic
        # if env.players[current_player].implied_knowledge[0].count(0) > 0: legal_conventions.append(9+self.environment_action_space)  #this and convention 12 might be ambigious

        # #10
        # # legal_conventions.append(0)
        # if self.hint_tokens > 0:
        #     other_player_card_1 = self.other_player_hand[0]
        #     other_player_card_2 = self.other_player_hand[1]
        #     if other_player_card_1%10 == other_player_card_2%10 or int(other_player_card_1/10) == int(other_player_card_2/10): legal_conventions.append(10+self.environment_action_space)

        # #11
        # if self.current_player_hinted[0] != 0: legal_conventions.append(11+self.environment_action_space)

        # #12
        # if env.players[current_player].implied_knowledge[0].count(0) > 0 or env.players[current_player].implied_knowledge[1].count(0) > 0: legal_conventions.append(12+self.environment_action_space)  #this and convention 12 might be ambigious

        return legal_conventions
    
    def make_env_usable(self, env):
        # This function extracts all the needed information from the current state and player observation to determine which conventions are available.
        # It receives the entire env as input to extract all the needed game features, but never gives an agent more information than it already
        # has access too. 

        current_player = env.state.cur_player()
        for player in range(env.state.num_players()):
            if player != current_player: other_player = player
        self.current_player = current_player
        self.other_player = other_player

        self.hint_tokens = env.state.information_tokens()

        fireworks_raw = env.state.fireworks()
        self.fireworks = fireworks_raw

        self.score = env.state.score()

        raw_hands = env.state.player_hands()

        current_player_hand = raw_hands[current_player]
        self.current_player_hand = current_player_hand
        card_counter = 0
        for card in current_player_hand:
            card_col = card._color
            card_val = card._rank
            self.current_player_hand[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        other_player_hand = raw_hands[other_player]
        self.other_player_hand = other_player_hand
        card_counter = 0
        for card in other_player_hand:
            card_col = card._color
            card_val = card._rank
            self.other_player_hand[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        raw_players_obs = env._make_observation_all_players()
        player_obs = raw_players_obs['player_observations'][current_player]['vectorized']
        legal_moves = raw_players_obs['player_observations'][current_player]['legal_moves_as_int']
        # self.legal_moves = format_legal_moves(legal_moves, env.num_moves())
        self.env_legal_moves = legal_moves

        raw_current_player_hinted = raw_players_obs['player_observations'][0]['card_knowledge'][current_player]
        raw_other_player_hinted = raw_players_obs['player_observations'][0]['card_knowledge'][other_player]

        self.current_player_hinted = []
        for hinted_card in raw_current_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.current_player_hinted.append(known)

        self.other_player_hinted = []
        for hinted_card in raw_other_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.other_player_hinted.append(known)

        raw_discard_pile = env.state.discard_pile()
        self.discard_pile = raw_discard_pile
        card_counter = 0
        for card in raw_discard_pile:
            card_col = card._color
            card_val = card._rank
            self.discard_pile[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        # pass #for debugging

class simple_transfer_encoder():
    # This function was an experimental attempt at transfer learning, where the agents have access to conventions until a certain number of steps, and then loose access to them.
    # It did not result in any performance gain, but was left in as an potential avenue for future research attempts.

    #-----------------List of conventions and their action ID-------------------
    #   0 (11) - discard oldest card, unless colour hinted. if both colour hinted, discard oldest
    #   1 (12)- hint next playable card (number)    
    #   2 (13)- play value card playable
    #   3 (14)- if score < 5, hint 5 to discard                     
    #   4 (15)- if 5 is hinted and score < 5, discard 5
    #   5 (16)- colour hint to keep card matching in discard pile and is not playable now
    #   6 (17)- hint value card to indicate discard
    #   7 (18)- discard value hinted card clearly not playable
    #-----------------------------------------------------------------------------------------
    def __init__(self, env):
        self.environment_action_space = env.num_moves()
        # self.convention_action_space = self.environment_action_space + 8
        self.convention_action_space = self.environment_action_space + 5

    def encode_action(self, agent_action, env):   #other player can be extracted from env using current player, must imp later
        if agent_action < self.environment_action_space:
            return agent_action
        
        else: agent_action -= self.environment_action_space

        self.make_env_usable(env)
        
        if agent_action == 0: 
            if int(self.current_player_hinted[0]/10) == 0: environment_action = 0
            elif int(self.current_player_hinted[0]/10) == 0 and int(self.current_player_hinted[1]/10) == 0: environment_action = 0
            elif int(self.current_player_hinted[0]/10) > 0 and int(self.current_player_hinted[1]/10) > 0: environment_action = 0
            else: environment_action = 1

        elif agent_action == 1:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    environment_action = 6 + (card%10)-1    #offsets action

        elif agent_action == 2:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10)-1 in current_firework_values:  #removes colour and decriments to match firework
                    environment_action = card_pos + 2

                card_pos += 1

        elif agent_action == 3:
            if self.score < 5:
                environment_action = 10

        elif agent_action == 4:
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10) == 5: 
                    environment_action = card_pos

                card_pos += 1

        # elif agent_action == 5:
        #     for card in self.other_player_hand:
        #         if card in self.discard_pile: environment_action = int(card/10) + 3 #colour hint to keep

        # elif agent_action == 6:
        #     #extract current fireworks 
        #     current_firework_values = self.fireworks
        #     #check which card to hint in other player hand following firework
        #     for card in self.other_player_hand:
        #         if card%10 <= current_firework_values[int(card/10)-1]:  #removes colour and decriments to match firework
        #             environment_action = 5 + (card%10)    #offsets action

        # elif agent_action == 7:
        #     #extract current fireworks 
        #     current_firework_values = self.fireworks
        #     #check which card to hint in other player hand following firework
        #     card_pos = 0
        #     for card in self.current_player_hinted:
        #         if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1]) and card%10 != 0:  #removes colour and decriments to match firework
        #             environment_action = card_pos

        #         card_pos += 1
            
        return environment_action
    
    def available_conventions(self, env, disable_encoding = False):
        legal_conventions = []      #positional arguments correlate to actions in description

        self.make_env_usable(env)

        #environment mask
        legal_conventions = legal_conventions + self.env_legal_moves

        current_firework_values = self.fireworks

        if not disable_encoding:
            #0
            # legal_conventions.append(1)
            if self.hint_tokens < 3: legal_conventions.append(0+self.environment_action_space)
            # else: legal_conventions.append(0)

            #1
            #check which card to hint in other player hand following firework
            if self.hint_tokens > 0:
                checker = False
                for card in self.other_player_hand:
                    if (card%10)-1 == current_firework_values[int(card/10) - 1]: checker = True

                if checker: legal_conventions.append(1+self.environment_action_space)
            # else: legal_conventions.append(0)

            #2
            checker = False
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10)-1 in current_firework_values: checker = True
                card_pos += 1

            if checker: legal_conventions.append(2+self.environment_action_space)

            #3
            if self.score < 5 and self.hint_tokens > 0:
                checker = False
                for card in self.other_player_hand:
                    if (card%10) == 5: checker = True

                if checker: legal_conventions.append(3+self.environment_action_space)
            # else: legal_conventions.append(0)

            #4
            if self.score < 5 and self.hint_tokens < 3:
                checker = False
                for card in self.current_player_hinted:
                    if (card%10) == 5: checker = True

                if checker: legal_conventions.append(4+self.environment_action_space)
            # else: legal_conventions.append(0)

            #5
            # if self.hint_tokens > 0:
            #     checker = False
            #     card_pos = 0
            #     for card in self.other_player_hand:
            #         if card in self.discard_pile and int(self.other_player_hinted[card_pos]/10) != 0: checker = True #colour hint to keep
            #         card_pos += 1
            #     if checker: legal_conventions.append(5+self.environment_action_space)
            # # else: legal_conventions.append(0)

            # #6
            # if self.hint_tokens > 0:
            #     checker = False
            #     for card in self.other_player_hand:
            #         if card%10 <= current_firework_values[int(card/10)-1]: checker = True

            #     if checker: legal_conventions.append(6+self.environment_action_space)
            # # else: legal_conventions.append(0)

            # #7
            # if self.hint_tokens < 3:
            #     checker = False
            #     card_pos = 0
            #     for card in self.current_player_hinted:
            #         if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1]) and card%10 != 0: checker = True
            #         card_pos += 1

            #     if checker: legal_conventions.append(7+self.environment_action_space)

        return legal_conventions
    
    def make_env_usable(self, env):
        current_player = env.state.cur_player()
        for player in range(env.state.num_players()):
            if player != current_player: other_player = player
        self.current_player = current_player
        self.other_player = other_player

        self.hint_tokens = env.state.information_tokens()

        fireworks_raw = env.state.fireworks()
        self.fireworks = fireworks_raw

        self.score = env.state.score()

        raw_hands = env.state.player_hands()

        current_player_hand = raw_hands[current_player]
        self.current_player_hand = current_player_hand
        card_counter = 0
        for card in current_player_hand:
            card_col = card._color
            card_val = card._rank
            self.current_player_hand[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        other_player_hand = raw_hands[other_player]
        self.other_player_hand = other_player_hand
        card_counter = 0
        for card in other_player_hand:
            card_col = card._color
            card_val = card._rank
            self.other_player_hand[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        raw_players_obs = env._make_observation_all_players()
        player_obs = raw_players_obs['player_observations'][current_player]['vectorized']
        legal_moves = raw_players_obs['player_observations'][current_player]['legal_moves_as_int']
        # self.legal_moves = format_legal_moves(legal_moves, env.num_moves())
        self.env_legal_moves = legal_moves

        raw_current_player_hinted = raw_players_obs['player_observations'][0]['card_knowledge'][current_player]
        raw_other_player_hinted = raw_players_obs['player_observations'][0]['card_knowledge'][other_player]

        self.current_player_hinted = []
        for hinted_card in raw_current_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.current_player_hinted.append(known)

        self.other_player_hinted = []
        for hinted_card in raw_other_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.other_player_hinted.append(known)

        raw_discard_pile = env.state.discard_pile()
        self.discard_pile = raw_discard_pile
        card_counter = 0
        for card in raw_discard_pile:
            card_col = card._color
            card_val = card._rank
            self.discard_pile[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        # pass #for debugging

if __name__ == '__main__':  # Used to check if the conventions are working correctly through debugging and human play
    game = rl_env.make(environment_name='Hanabi-Small', num_players=2, pyhanabi_path=None)
    # convention_encoder = standalone_encoder(game)
    convention_encoder = simple_combined_encoder(game)

    for eps in range(10):
        done = False
        observations = game.reset()

        print('Press enter for next frame')
        ui = input()

        while not done:
            print(game.state)
            # print(game.state.cur_player())
            # print(game.state.observation(game.state.cur_player()))
            # convention_encoder.make_env_usable(game)

            # print(f"legal moves: {convention_encoder.available_conventions(active_player, other_player, game)}")
            # print(f"legal moves: {convention_encoder.env_legal_moves}")
            legal_convention_moves = convention_encoder.available_conventions(game)
            print(f"legal moves with conventions: {legal_convention_moves}")
            print(f"vectorized conventions:{format_legal_moves(legal_convention_moves, convention_encoder.convention_action_space)}")
            while(ui == ""):
                print('Enter action')
                ui = input()

            env_action = convention_encoder.encode_action(int(ui), game)

            # observations, reward, is_done, _ = game.step(int(ui))
            observations, reward, is_done, _ = game.step(env_action)
            
            print('Press enter for next frame')
            ui = input()

        game.reset()
    
