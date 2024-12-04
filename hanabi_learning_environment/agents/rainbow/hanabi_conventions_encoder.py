#   Orignal author: F Bredell
#   Last modified: 25 November 2024
#
#   This code serves as the conventions extraction and translation layers for the agent network. 
#   All conventions are stated before the init function and each class contains their own env processing layer 
#   to make the observation usable with the conventions. 
#
#   It is strongly recommended to fimiliarise yourself with the current Hanabi learning environment and how the action space is currently set up for all player counts. I recommend
#   making a list of 0-max env action with notes on what each action does, then append to that list the conventions as described in each functions heading. Also read through the
#   basic conventions defined by H-group and make sure you do their 'quizez' to understand some of the more niche and tricky parts of each convention.

import numpy as np
from hanabi_learning_environment import rl_env

def format_legal_moves(legal_moves, action_dim):
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
    
class simple_combined_encoder_full():
    #   Initial exploration of conventions as options specific to Hanabi, however these conventions were in-house and non-optimal. It is used in tandem with primitive actions to
    #   form action augmentation 
    #
    #-----------------List of conventions and their action ID-------------------
    #   0 (11) - discard oldest card, unless something is hinted. if all have something hinted, discard oldest none fully revealed
    #   1 (12)- hint next playable card (number)                
    #   2 (13)- play value card playable
    #   3 (14)- hint next playable card (colour)    
    #   4 (15)- play newest colour card hinted  #stops here for now
    #   5 (16)- colour hint to keep card matching in discard pile and is not playable now
    #   6 (17)- hint value card to indicate discard
    #   7 (18)- discard value hinted card clearly not playable
    #---------------------------------------------------------------------------------

    def __init__(self, env):
        self.environment_action_space = env.num_moves()
        # self.convention_action_space = self.environment_action_space + 8  #with optional extras
        self.convention_action_space = self.environment_action_space + 5

    def encode_action(self, agent_action, env):
        # This function is the output layer after an agent chooses a convention and before the action is sent to the environment. 
        # It translates the chosen action/convention into an environment action based on the convention principles.

        if agent_action < self.environment_action_space:
            return agent_action
        
        else: agent_action -= self.environment_action_space

        self.make_env_usable(env)
        
        if agent_action == 0: 
            # if int(self.current_player_hinted[0]/10) == 0: environment_action = 0
            # elif int(self.current_player_hinted[0]/10) == 0 and int(self.current_player_hinted[1]/10) == 0: environment_action = 0
            # elif int(self.current_player_hinted[0]/10) > 0 and int(self.current_player_hinted[1]/10) > 0: environment_action = 0
            # else: environment_action = 1
            counter = 0
            found_a_card = False
            for card in self.current_player_hinted:
                if card == 0: 
                    found_a_card = True
                    break
                counter += 1

            if not found_a_card: 
                counter = 0
                for card in self.current_player_hinted:
                    if int(card/10) == 0 or card%10 == 0: 
                        found_a_card = True
                        break
                    counter += 1
            
            if not found_a_card: environment_action = 0

            environment_action = counter

        elif agent_action == 1:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    environment_action = 15 + (card%10)-1    #offsets action

        elif agent_action == 2:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10)-1 in current_firework_values:  #removes colour and decriments to match firework
                    environment_action = card_pos + 5

                card_pos += 1

        elif agent_action == 3:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    environment_action = 10 + int(card/10)-1    #offsets action

        elif agent_action == 4:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if int(card/10) != 0:  #removes colour and decriments to match firework
                    environment_action = card_pos + 5

                card_pos += 1

        #-------------------------------------------Optional extras--------------------------------------------
        # elif agent_action == 5:
        #     for card in self.other_player_hand:
        #         if card in self.discard_pile: environment_action = int(card/10)-1 + 10 #colour hint to keep

        # elif agent_action == 6:
        #     #extract current fireworks 
        #     current_firework_values = self.fireworks
        #     #check which card to hint in other player hand following firework
        #     for card in self.other_player_hand:
        #         if card%10 <= current_firework_values[int(card/10)-1]:  #removes colour and decriments to match firework
        #             environment_action = 14 + (card%10)    #offsets action

        # elif agent_action == 7:
        #     #extract current fireworks 
        #     current_firework_values = self.fireworks
        #     #check which card to hint in other player hand following firework
        #     card_pos = 0
        #     for card in self.current_player_hinted:
        #         if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1] or card%10 <= current_firework_values[2] or card%10 <= current_firework_values[3] or card%10 <= current_firework_values[4]) and card%10 != 0:  #removes colour and decriments to match firework
        #             environment_action = card_pos

        #         card_pos += 1
        #--------------------------------------------------------------------------------------------------------
            
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
        if self.hint_tokens < 8: legal_conventions.append(0+self.environment_action_space)
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
        #check which card to hint in other player hand following firework
        if self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]: checker = True

            if checker: legal_conventions.append(3+self.environment_action_space)
        # else: legal_conventions.append(0)

        #4
        checker = False
        card_pos = 0
        for card in self.current_player_hinted:
            if int(card/10) != 0: checker = True
            card_pos += 1

        if checker: legal_conventions.append(4+self.environment_action_space)

        #-------------------------------------------Optional extras--------------------------------------------
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
        #         if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1] or card%10 <= current_firework_values[2] or card%10 <= current_firework_values[3] or card%10 <= current_firework_values[4]) and card%10 != 0: checker = True
        #         card_pos += 1

        #     if checker: legal_conventions.append(7+self.environment_action_space)
        #--------------------------------------------------------------------------------------------------------

        return legal_conventions
    
    def make_env_usable(self, env):
        # This function extracts all the needed information from the current state and player observation to determine which conventions are available.
        # It receives the entire env as input to extract all the needed game features, but never gives an agent more information than it already
        # has access too. 

        current_player = env.state.cur_player()
        for player in range(env.state.num_players()):
            if player != current_player: other_player = player      #only works for 2 player, must be modded for more
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
                elif hinted_card['color'] == 'G': known += 30
                elif hinted_card['color'] == 'W': known += 40
                elif hinted_card['color'] == 'B': known += 50
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.current_player_hinted.append(known)

        self.other_player_hinted = []
        for hinted_card in raw_other_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
                elif hinted_card['color'] == 'G': known += 30
                elif hinted_card['color'] == 'W': known += 40
                elif hinted_card['color'] == 'B': known += 50
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

class simple_combined_encoder_full_v2():
    # revised version of above encoder with bugs removed and improved conventions
    #
    #-----------------List of conventions and their action ID-------------------
    #   0 (11) - discard oldest card, unless colour hinted. if both colour hinted, discard oldest
    #   1 (12)- hint next playable card (number)    
    #   2 (13)- play value card playable
    #   3 (14)- colour hint to keep card matching in discard pile and is not playable now
    #   4 (15)- hint value card to indicate discard
    #   5 (16)- discard value hinted card clearly not playable
    #----------------------------------------------------------------------------------------

    def __init__(self, env):
        self.environment_action_space = env.num_moves()
        # self.convention_action_space = self.environment_action_space + 8
        self.convention_action_space = self.environment_action_space + 6

    def encode_action(self, agent_action, env):  
        if agent_action < self.environment_action_space:
            return agent_action
        
        else: agent_action -= self.environment_action_space

        self.make_env_usable(env)
        
        if agent_action == 0: 
            # if int(self.current_player_hinted[0]/10) == 0: environment_action = 0
            # elif int(self.current_player_hinted[0]/10) == 0 and int(self.current_player_hinted[1]/10) == 0: environment_action = 0
            # elif int(self.current_player_hinted[0]/10) > 0 and int(self.current_player_hinted[1]/10) > 0: environment_action = 0
            # else: environment_action = 1
            counter = 0
            found_a_card = False
            for card in self.current_player_hinted:
                if card == 0: 
                    found_a_card = True
                    break
                counter += 1

            if not found_a_card: 
                counter = 0
                for card in self.current_player_hinted:
                    if int(card/10) == 0 == 0: 
                        found_a_card = True
                        break
                    counter += 1
            
            if not found_a_card: environment_action = 0

            environment_action = counter

        elif agent_action == 1:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    environment_action = 15 + (card%10)-1    #offsets action

        elif agent_action == 2:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10)-1 in current_firework_values:  #removes colour and decriments to match firework
                    environment_action = card_pos + 5

                card_pos += 1


        elif agent_action == 3:
            for card in self.other_player_hand:
                if card in self.discard_pile: environment_action = int(card/10)-1 + 10 #colour hint to keep

        elif agent_action == 4:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            for card in self.other_player_hand:
                if card%10 <= current_firework_values[int(card/10)-1]:  #removes colour and decriments to match firework
                    environment_action = 15 + (card%10)-1    #offsets action

        elif agent_action == 5:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1] or card%10 <= current_firework_values[2] or card%10 <= current_firework_values[3] or card%10 <= current_firework_values[4]) and card%10 != 0:  #removes colour and decriments to match firework
                    environment_action = card_pos

                card_pos += 1
            
        return environment_action
    
    def available_conventions(self, env):
        legal_conventions = []      #positional arguments correlate to actions in description

        self.make_env_usable(env)

        #environment mask
        legal_conventions = legal_conventions + self.env_legal_moves

        current_firework_values = self.fireworks

        #0
        # legal_conventions.append(1)
        if self.hint_tokens < 8: legal_conventions.append(0+self.environment_action_space)
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
        if self.hint_tokens > 0:
            checker = False
            card_pos = 0
            for card in self.other_player_hand:
                if card in self.discard_pile and int(self.other_player_hinted[card_pos]/10) != 0: checker = True #colour hint to keep
                card_pos += 1
            if checker: legal_conventions.append(3+self.environment_action_space)
        # else: legal_conventions.append(0)

        #4
        if self.hint_tokens > 0:
            checker = False
            for card in self.other_player_hand:
                if card%10 <= current_firework_values[int(card/10)-1]: checker = True

            if checker: legal_conventions.append(4+self.environment_action_space)
        # else: legal_conventions.append(0)

        #5
        if self.hint_tokens < 3:
            checker = False
            card_pos = 0
            for card in self.current_player_hinted:
                if (card%10 <= current_firework_values[0] or card%10 <= current_firework_values[1] or card%10 <= current_firework_values[2] or card%10 <= current_firework_values[3] or card%10 <= current_firework_values[4]) and card%10 != 0: checker = True
                card_pos += 1

            if checker: legal_conventions.append(5+self.environment_action_space)

        return legal_conventions
    
    def make_env_usable(self, env):
        current_player = env.state.cur_player()
        for player in range(env.state.num_players()):
            if player != current_player: other_player = player      #only works for 2 player, must be modded for more
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
                elif hinted_card['color'] == 'G': known += 30
                elif hinted_card['color'] == 'W': known += 40
                elif hinted_card['color'] == 'B': known += 50
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.current_player_hinted.append(known)

        self.other_player_hinted = []
        for hinted_card in raw_other_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
                elif hinted_card['color'] == 'G': known += 30
                elif hinted_card['color'] == 'W': known += 40
                elif hinted_card['color'] == 'B': known += 50
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

class simple_official_rules_based_encoder_2p():
    # Conventions encoder based on the Hanabi conventions defined by H-Group. This implementation only works for 2 player and was the first iteration of the conventions encoder. 
    #
    #-----------------Things to extract--------------------------------
    # Chop card - oldest card in hand that has no hints on it
    # Round actions - all the actions in the current round
    #------------------------------------------------------------------
    #
    #-----------------List of conventions and their action ID-------------------
    #   0 - play-hint colour card in next player's hand (for more than 2 players it might be beneficial to seperate each player as a convention to hint)
    #   1 - play-hint value card in next player's hand (for more than 2 players it might be beneficial to seperate each player as a convention to hint)
    #   2 - play colour hinted card in hand based on actions this round, if more than one card is hinted then always play newest card
    #   3 - play value hinted card in hand based on actions this round, if more than one card is hinted then always play newest card
    #   4 - if chop is a 5, value 5 save hint to player
    #   5 - if chop is a unique 2 based on current player's perspective, then value save hint 2
    #   6 - if chop is last one of its kind, critical save chop with either value of colour hint based on other cards in player's hand
    #   7 - prompt, if 2 player then self prompt
    #   8 - react to prompt clue
    #   9 - finesse, if 2 player then cannot
    #   10 - react to finesse, if 2 player then cannot
    #   11 - discard card in chop position, if there is no chop then you cannot discard. Also cannot discard in early game, i.e. if other conventions are still doable and the
    #   discard pile is empty, then you cannot do this.
    #-------------------------------------------------------------------------------------

    def __init__(self, env):
        self.environment_action_space = env.num_moves()
        self.convention_action_space = self.environment_action_space + 12 #number of conventions starting at 0

        self.previous_action = 0     

    def reset(self):
        self.previous_action = 0     

    def encode_action(self, agent_action, env):   
        # This function is the output layer after an agent chooses a convention and before the action is sent to the environment. 
        # It translates the chosen action/convention into an environment action based on the convention principles.

        if agent_action < self.environment_action_space:
            self.previous_action = agent_action
            return agent_action
        
        else: agent_action -= self.environment_action_space

        self.make_env_usable(env)

        if agent_action == 0:       
            current_firework_values = self.fireworks
            card_counter = 0
            for card in self.other_player_hand:
                ambig_found = False
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    other_card_counter = 0
                    for other_card in self.other_player_hand:
                        if other_card_counter == card_counter:
                            pass

                        elif other_card%10 == card%10 and other_card_counter > card_counter and card_counter != self.other_player_chop_position:
                            ambig_found = True

                        elif other_card%10 == card%10 and other_card_counter == self.other_player_chop_position:
                            ambig_found = True

                        other_card_counter += 1
                    
                    if self.other_player_hand.count(card) == 1 and not ambig_found:     #we do not hint doubles since it gives useless info
                        environment_action = 15 + (card%10)-1
                        break

                card_counter += 1

        elif agent_action == 1:
            #extract current fireworks 
            current_firework_values = self.fireworks
            #check which card to hint in other player hand following firework
            card_counter = 0
            for card in self.other_player_hand:
                ambig_found = False
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    other_card_counter = 0
                    for other_card in self.other_player_hand:
                        if other_card_counter == card_counter:
                            pass

                        elif int(other_card/10) == int(card/10) and other_card%10 > card%10 and other_card_counter > card_counter and card_counter != self.other_player_chop_position:
                            ambig_found = True

                        elif int(other_card/10) == int(card/10) and other_card_counter == self.other_player_chop_position:
                            ambig_found = True

                        other_card_counter += 1

                    if not ambig_found:
                        environment_action = 10 + int(card/10) - 1    #offsets action
                        break

                card_counter += 1        
                    

        elif agent_action == 2:
            if self.previous_action >= 10 and self.previous_action < 15:
                hinted_colour = (self.previous_action - 10) + 1
                card_was_before_current_chop = False
                card_position_before_chop = None

                counter = 0
                for hinted_card in self.current_player_hinted:  #checks if the focus might have been on the chop card, which takes precidence over non chop cards NB
                    if int(hinted_card/10) == hinted_colour and counter < self.current_player_chop_position: 
                        if int(hinted_card/10) != 0 and hinted_card%10 != 0: pass    #both values can only be known if chop did not move, i.e. a check to see if chopped moved because of hint
                        else: card_was_before_current_chop = True

                        card_position_before_chop = counter
                    counter += 1

                if card_was_before_current_chop and (self.current_player_chop_position - card_position_before_chop) == 1:
                    environment_action = card_position_before_chop + 5
                else:
                    #check which card was hinted and play the newest one if none of them are the previous chop card NB!
                    card_pos = 0
                    for card in self.current_player_hinted:
                        if int(card/10) == hinted_colour:  #removes colour and decriments to match firework
                            environment_action = card_pos + 5

                        card_pos += 1

        elif agent_action == 3:
            if self.previous_action >= 15 and self.previous_action < 20:
                hinted_value = (self.previous_action - 15) + 1
                card_was_before_current_chop = False
                card_position_before_chop = None

                counter = 0
                for hinted_card in self.current_player_hinted:  #checks if the focus might have been on the chop card, which takes precidence over non chop cards NB
                    if hinted_card%10 == hinted_value and counter < self.current_player_chop_position: 
                        if int(hinted_card/10) != 0 and hinted_card%10 != 0: pass    #both values can only be known if chop did not move, i.e. a check to see if chopped moved because of hint
                        else: card_was_before_current_chop = True
                        card_position_before_chop = counter

                    counter += 1

                if card_was_before_current_chop and (self.current_player_chop_position - card_position_before_chop) == 1:
                    environment_action = card_position_before_chop + 5
                else:
                    #check which card was hinted and play the newest one if none of them are the previous chop card NB!
                    card_pos = 0
                    for card in self.current_player_hinted:
                        if card%10 == hinted_value:  #removes colour and decriments to match firework
                            environment_action = card_pos + 5

                        card_pos += 1

        elif agent_action == 4:
            if self.other_player_hand[self.other_player_chop_position]%10 == 5:
                environment_action = 19

        elif agent_action == 5:
            if self.other_player_hand.count(self.other_player_hand[self.other_player_chop_position]) == 1 and self.other_player_hand[self.other_player_chop_position]%10 == 2:
                environment_action = 16

        elif agent_action == 6:
            if self.other_player_hand[self.other_player_chop_position]%10 == 1 and self.discard_pile.count(self.other_player_hand[self.other_player_chop_position]) == 2:
                card_value = self.other_player_hand[self.other_player_chop_position]%10
                card_colour = int(self.other_player_hand[self.other_player_chop_position]/10)

                unique_colour = True
                unique_value = True
                counter = 0
                for card in self.other_player_hand:
                    if counter != self.other_player_chop_position:
                        if card%10 == card_value: unique_value = False
                        if int(card/10) == card_colour: unique_colour = False

                    counter += 1

                if unique_colour: environment_action = 10 + card_colour-1
                elif unique_value: environment_action = 15 + card_value-1
                else: environment_action = 10 + card_colour-1


            elif self.other_player_hand[self.other_player_chop_position] in self.discard_pile:
                card_value = self.other_player_hand[self.other_player_chop_position]%10
                card_colour = int(self.other_player_hand[self.other_player_chop_position]/10)

                unique_colour = True
                unique_value = True
                counter = 0
                for card in self.other_player_hand:
                    if counter != self.other_player_chop_position:
                        if card%10 == card_value: unique_value = False
                        if int(card/10) == card_colour: unique_colour = False

                    counter += 1

                if unique_colour: environment_action = 10 + card_colour-1
                elif unique_value: environment_action = 15 + card_value-1
                else: environment_action = 10 + card_colour-1

        elif agent_action == 7:
            current_firework_values = self.fireworks
            card_pos = 0
            for card in self.other_player_hand:
                if (card%10)-1 == current_firework_values[int(card/10) - 1] and self.other_player_hinted[card_pos]%10 != 0:  #removes colour and decriments to match firework
                    for other_card in self.other_player_hand:
                        if other_card%10 == card%10 + 1 and int(other_card/10) == int(card/10): 
                            environment_action = (other_card%10 - 1) + 15

                card_pos += 1

        elif agent_action == 8:
            if self.previous_action >= 15 and self.previous_action < 20:
                hinted_value = self.previous_action - 15 + 1
                card_pos = 0
                for card in self.current_player_hinted:
                    if (card%10) != 0 and card%10 == hinted_value - 1:
                        environment_action = card_pos + 5
                        
                    card_pos += 1

        elif agent_action == 9:
            pass

        elif agent_action == 10:
            pass

        elif agent_action == 11: 
            environment_action = self.current_player_chop_position
            
        self.previous_action = environment_action

        return environment_action
    
    def available_conventions(self, env):
        # This function is at the input of the network to extract the current available conventions based on the current observation of a player. 
        # It also incorporates action masking and formatting to allow for the Dopamine agent to interpret it correctly.

        legal_conventions = []      #positional arguments correlate to actions in description

        self.make_env_usable(env)

        #environment mask
        legal_conventions = legal_conventions + self.env_legal_moves

        current_firework_values = self.fireworks
        early_game_checker = False

        #0
        if self.hint_tokens > 0:            
            card_counter = 0
            for card in self.other_player_hand:
                ambig_found = False
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    other_card_counter = 0
                    for other_card in self.other_player_hand:
                        if other_card_counter == card_counter:
                            pass

                        elif other_card%10 == card%10 and other_card_counter > card_counter and card_counter != self.other_player_chop_position:
                            ambig_found = True

                        elif other_card%10 == card%10 and other_card_counter == self.other_player_chop_position:
                            ambig_found = True

                        other_card_counter += 1
                    
                    if self.other_player_hand.count(card) == 1 and not ambig_found:     #we do not hint doubles since it gives useless info
                        legal_conventions.append(0+self.environment_action_space)
                        early_game_checker = True
                        break

                card_counter += 1

        #1
        if self.hint_tokens > 0:
            card_counter = 0
            for card in self.other_player_hand:
                ambig_found = False
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    other_card_counter = 0
                    for other_card in self.other_player_hand:
                        if other_card_counter == card_counter:
                            pass

                        elif int(other_card/10) == int(card/10) and other_card%10 > card%10 and other_card_counter > card_counter and card_counter != self.other_player_chop_position:
                            ambig_found = True

                        elif int(other_card/10) == int(card/10) and other_card%10 <= current_firework_values[int(card/10)-1] and other_card_counter > card_counter and card_counter != self.other_player_chop_position:
                            ambig_found = True

                        elif int(other_card/10) == int(card/10) and other_card_counter == self.other_player_chop_position:
                            ambig_found = True

                        other_card_counter += 1
                    
                    if self.other_player_hand.count(card) == 1 and not ambig_found:     #we do not hint doubles since it gives useless info, also avoid ambigious hints
                        legal_conventions.append(1+self.environment_action_space)
                        early_game_checker = True
                        break

                card_counter += 1

        #2
        if self.previous_action >= 10 and self.previous_action < 15: 
            hinted_colour = self.previous_action - 10 + 1
            for card in self.current_player_hinted:
                if int(card/10) == hinted_colour: 
                    legal_conventions.append(2+self.environment_action_space)
                    early_game_checker = True
                    break

                else: pass
        
        #3
        if self.previous_action >= 15 and self.previous_action < 20: 
            hinted_value = self.previous_action - 15 + 1
            for card in self.current_player_hinted:
                if (card%10) == hinted_value and (card%10) - 1 in current_firework_values:
                    legal_conventions.append(3+self.environment_action_space)
                    early_game_checker = True
                    break

                else: pass

        #4
        if self.hint_tokens > 0 and self.other_player_chop_position < 5:
            if self.other_player_hand[self.other_player_chop_position]%10 == 5:
                legal_conventions.append(4+self.environment_action_space)
                early_game_checker = True

        #5
        if self.hint_tokens > 0 and self.other_player_chop_position < 5:
            if (self.other_player_hand.count(self.other_player_hand[self.other_player_chop_position]) == 1 and self.other_player_hand[self.other_player_chop_position]%10 == 2) and self.other_player_hand[self.other_player_chop_position]%10 > current_firework_values[int(self.other_player_hand[self.other_player_chop_position]/10) - 1]:
                legal_conventions.append(5+self.environment_action_space)
                early_game_checker = True

        #6
        if self.hint_tokens > 0 and self.other_player_chop_position < 5:
            if self.other_player_hand[self.other_player_chop_position]%10 == 1 and self.discard_pile.count(self.other_player_hand[self.other_player_chop_position]) == 2 and self.other_player_hand[self.other_player_chop_position]%10 > current_firework_values[int(self.other_player_hand[self.other_player_chop_position]/10) - 1]:
                legal_conventions.append(6+self.environment_action_space)
                early_game_checker = True

            elif self.other_player_hand[self.other_player_chop_position] in self.discard_pile and self.other_player_hand[self.other_player_chop_position]%10 > current_firework_values[int(self.other_player_hand[self.other_player_chop_position]/10) - 1] and self.other_player_hand[self.other_player_chop_position]%10 != 1:
                legal_conventions.append(6+self.environment_action_space)
                early_game_checker = True

        #7                              #can also try disabling this for 2 players since it might be causing issues... it was
        # if self.hint_tokens > 0:
        #     card_pos = 0
        #     for card in self.other_player_hand:
        #         if (card%10)-1 == current_firework_values[int(card/10) - 1] and self.other_player_hinted[card_pos]%10 != 0:  #removes colour and decriments to match firework
        #             for other_card in self.other_player_hand:
        #                 if other_card%10 == card%10 + 1 and int(other_card/10) == int(card/10): 
        #                     legal_conventions.append(7+self.environment_action_space)
        #                     early_game_checker = True

        #         card_pos += 1

        if env.state.num_players() == 2:
            pass

        #8                              #can also try disabling this for 2 players since it might be causing issues... it was
        # if self.previous_action >= 15 and self.previous_action < 20:
        #     hinted_value = self.previous_action - 15 + 1
        #     for card in self.current_player_hinted:
        #         if (card%10) != 0 and card%10 == hinted_value - 1:
        #             legal_conventions.append(8+self.environment_action_space)
        #             early_game_checker = True

        if env.state.num_players() == 2:
            pass

        #9
        if env.state.num_players() == 2:
            pass

        #10
        if env.state.num_players() == 2:
            pass

        #11
        if ((not self.discard_pile) and early_game_checker) and self.hint_tokens != 0: #check for early game
            pass
        else:
            if self.hint_tokens < 8:
                if self.current_player_chop_position < 5: legal_conventions.append(11+ self.environment_action_space)

        return legal_conventions
    
    def make_env_usable(self, env):
        # This function extracts all the needed information from the current state and player observation to determine which conventions are available.
        # It receives the entire env as input to extract all the needed game features, but never gives an agent more information than it already
        # has access too. 

        current_player = env.state.cur_player()
        for player in range(env.state.num_players()):
            if player != current_player: other_player = player      #only works for 2 player, next class works for all player counts
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
                elif hinted_card['color'] == 'G': known += 30
                elif hinted_card['color'] == 'W': known += 40
                elif hinted_card['color'] == 'B': known += 50
            if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

            self.current_player_hinted.append(known)

        self.other_player_hinted = []
        for hinted_card in raw_other_player_hinted:
            known = 0
            if hinted_card['color'] != None: 
                if hinted_card['color'] == 'R': known += 10
                elif hinted_card['color'] == 'Y': known += 20
                elif hinted_card['color'] == 'G': known += 30
                elif hinted_card['color'] == 'W': known += 40
                elif hinted_card['color'] == 'B': known += 50
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

        self.current_player_chop_position = 5
        self.other_player_chop_position = 5

        counter = 0
        found_a_card = False
        for card in self.current_player_hinted:
            if card == 0:
                found_a_card = True
                break    
            counter += 1

        if found_a_card: self.current_player_chop_position = counter

        counter = 0
        found_a_card = False
        for card in self.other_player_hinted:
            if card == 0:
                found_a_card = True
                break    
            counter += 1

        if found_a_card: self.other_player_chop_position = counter

        # pass #for debugging

class simple_official_rules_based_encoder():
    # Final version of the rules based conventions encoder. It works for all player counts (2--5) and is based on the conventions and principles defined by H-group. 
    #
    #-----------------Things to extract--------------------------------
    # Chop card - oldest card in hand that has no hints on it
    # Round actions - all the actions in the current round
    #------------------------------------------------------------------------
    #
    #-----------------List of conventions and their action ID-------------------
    #   0 - play-hint value card in next player's hand (for more than 2 players it might be beneficial to seperate each player as a convention to hint)
    #   1 - play-hint colour card in next player's hand (for more than 2 players it might be beneficial to seperate each player as a convention to hint)
    #   2 - play value hinted card in hand based on actions this round, if more than one card is hinted then always play newest card
    #   3 - play colour hinted card in hand based on actions this round, if more than one card is hinted then always play newest card
    #   4 - if chop is a 5, value 5 save hint to player
    #   5 - if chop is a unique 2 based on current player's perspective, then value save hint 2
    #   6 - if chop is last one of its kind, critical save chop with either value of colour hint based on other cards in player's hand
    #   7 - prompt, for ease of implimenting, we only focus on consecutive player promts (in order of turns) and not out of sync promts since this requires to high level of ToM
    #   8 - react to prompt clue
    #   9 - finesse, if 2 player then cannot. Also only consec players following current player, since finesse causes momentary desynch of info
    #   10 - react to finesse, if 2 player then cannot
    #   11 - discard card in chop position, if there is no chop then you cannot discard. Also cannot discard in early game, i.e. if other conventions are still doable and the
    #   discard pile is empty, then you cannot do this.
    #-------------------------------------------------------------------------------
    #
    # When moving to more than two players the conventions are placed in groups according to which player they are applicable, for example, in a 3-player game convention 0 will now
    # have two elements; one for play-hint to next player and one to play-hint to next-next player, and so forth. In 4-player convention 0 will have 3 elements and so on. 
    # 

    def __init__(self, env):
        self.number_of_players = env.players
        self.environment_action_space = env.num_moves()
        self.convention_total = (7*(self.number_of_players-1))+5 # number of conventions starting at 0, and taking into account the growing conventions list as more players are added. 
        self.convention_action_space = self.environment_action_space + self.convention_total 

        self.previous_action = 0     
        self.previous_actions = [0] * (self.number_of_players-1)  #for more than 2p

        if self.number_of_players < 4: self.player_hand_size = 5
        else: self.player_hand_size = 4

        self.distribution_length = 5

    def reset(self):
        self.previous_action = 0     
        self.previous_actions = [0] * (self.number_of_players-1)  #for more than 2p

    def encode_action(self, agent_action, env): 
        # This function is the output layer after an agent chooses a convention and before the action is sent to the environment. 
        # It translates the chosen action/convention into an environment action based on the convention principles. 

        if agent_action < self.environment_action_space:  
            self.previous_action = agent_action
            self.previous_actions.append(agent_action)  # this is very important to capture for moves like the finesse and prompt
            if len(self.previous_actions) > self.number_of_players-1:self.previous_actions.pop(0)
            return agent_action
    
        else: agent_action -= self.environment_action_space 

        self.make_env_usable(env)

        offset_counter = 0  #   The offset counter is used to make sure the conventions translate correctly to the correct number of players, and to account for the conventions being splitted amongst more players

        #0
        if agent_action >= (offset_counter+0) and agent_action < (0 + offset_counter + self.number_of_players-1):   
            target_player_offset = agent_action-offset_counter-0+1    
            target_player = self.current_player + target_player_offset
            if target_player > max(self.all_players): target_player -= self.number_of_players
            current_firework_values = self.fireworks
            card_counter = 0
            for card in self.player_hands[target_player]:
                ambig_found = False
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    other_card_counter = 0
                    for other_card in self.player_hands[target_player]:
                        if other_card_counter == card_counter:
                            pass

                        elif other_card%10 == card%10 and other_card_counter > card_counter and card_counter != self.players_chop_positions[target_player]:
                            ambig_found = True

                        elif other_card%10 == card%10 and other_card_counter == self.players_chop_positions[target_player]:
                            ambig_found = True

                        other_card_counter += 1
                    
                    if self.player_hands[target_player].count(card) == 1 and not ambig_found:     #we do not hint doubles since it gives useless info
                        environment_action = (self.player_hand_size)*2 + (self.number_of_players-2+target_player_offset)*self.distribution_length + (card%10)-1
                        break

                card_counter += 1

        offset_counter += (self.number_of_players-2)
        #1
        if agent_action >= (offset_counter + 1) and agent_action < (1 + offset_counter + self.number_of_players-1):
            target_player_offset = agent_action-offset_counter-1+1
            target_player = self.current_player + target_player_offset
            if target_player > max(self.all_players): target_player -= self.number_of_players

            current_firework_values = self.fireworks

            card_counter = 0
            for card in self.player_hands[target_player]:
                ambig_found = False
                if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                    other_card_counter = 0
                    for other_card in self.player_hands[target_player]:
                        if other_card_counter == card_counter:
                            pass

                        elif int(other_card/10) == int(card/10) and other_card%10 > card%10 and other_card_counter > card_counter and card_counter != self.players_chop_positions[target_player]:
                            ambig_found = True

                        elif int(other_card/10) == int(card/10) and other_card_counter == self.players_chop_positions[target_player]:
                            ambig_found = True

                        other_card_counter += 1

                    if not ambig_found:
                        environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + int(card/10) - 1    #offsets action
                        break

                card_counter += 1        

        offset_counter += (self.number_of_players-2)
        #2
        if agent_action >= (offset_counter + 2) and agent_action < (2 + offset_counter + self.number_of_players-1):
            action_identify_counter = len(self.previous_actions)-1
            for action_to_act_on in self.previous_actions:
                if (agent_action - (offset_counter + 2 + action_identify_counter)) == 0: break
                action_identify_counter -= 1

            hinted_value = (action_to_act_on - ((self.player_hand_size)*2 + (self.number_of_players-2+action_identify_counter+1)*self.distribution_length)) + 1
            card_was_before_current_chop = False
            card_position_before_chop = None

            counter = 0
            for hinted_card in self.current_player_hinted:  #checks if the focus might have been on the chop card, which takes precidence over non chop cards NB
                if hinted_card%10 == hinted_value and counter < self.current_player_chop_position: 
                    if int(hinted_card/10) != 0 and hinted_card%10 != 0: pass    #both values can only be known if chop did not move, i.e. a check to see if chopped moved because of hint
                    else: card_was_before_current_chop = True
                    card_position_before_chop = counter

                counter += 1

            if card_was_before_current_chop and (self.current_player_chop_position - card_position_before_chop) == 1:
                environment_action = card_position_before_chop + self.player_hand_size
            else:
                #check which card was hinted and play the newest one if none of them are the previous chop card NB!
                card_pos = 0
                for card in self.current_player_hinted:
                    if card%10 == hinted_value:  #removes colour and decriments to match firework
                        environment_action = card_pos + self.player_hand_size

                    card_pos += 1

        offset_counter += (self.number_of_players-2)
        #3
        if agent_action >= (offset_counter + 3) and agent_action < (3 + offset_counter + self.number_of_players-1):
            action_identify_counter = len(self.previous_actions)-1
            for action_to_act_on in self.previous_actions:
                if (agent_action - (offset_counter + 3 + action_identify_counter)) == 0: break
                action_identify_counter -= 1

            hinted_colour = (action_to_act_on - ((self.player_hand_size)*2 + (action_identify_counter)*self.distribution_length)) + 1
            card_was_before_current_chop = False
            card_position_before_chop = None

            counter = 0
            for hinted_card in self.current_player_hinted:  #checks if the focus might have been on the chop card, which takes precidence over non chop cards NB
                if int(hinted_card/10) == hinted_colour and counter < self.current_player_chop_position: 
                    if int(hinted_card/10) != 0 and hinted_card%10 != 0: pass    #both values can only be known if chop did not move, i.e. a check to see if chopped moved because of hint
                    else: card_was_before_current_chop = True

                    card_position_before_chop = counter
                counter += 1

            if card_was_before_current_chop and (self.current_player_chop_position - card_position_before_chop) == 1:
                environment_action = card_position_before_chop + self.player_hand_size
            else:
                #check which card was hinted and play the newest one if none of them are the previous chop card NB!
                card_pos = 0
                for card in self.current_player_hinted:
                    if int(card/10) == hinted_colour:  #removes colour and decriments to match firework
                        environment_action = card_pos + self.player_hand_size

                    card_pos += 1

        offset_counter += (self.number_of_players-2)
        #4
        if agent_action >= (offset_counter + 4) and agent_action < (4 + offset_counter + self.number_of_players-1):
            target_player_offset = agent_action-offset_counter-4+1
            target_player = self.current_player + target_player_offset
            if target_player > max(self.all_players): target_player -= self.number_of_players

            if self.player_hands[target_player][self.players_chop_positions[target_player]]%10 == 5:
                environment_action = (self.player_hand_size)*2 + (self.number_of_players-2+target_player_offset)*self.distribution_length + 5-1

        offset_counter += (self.number_of_players-2)
        #5
        if agent_action >= (offset_counter + 5) and agent_action < (5 + offset_counter + self.number_of_players-1):
            target_player_offset = agent_action-offset_counter-5+1
            target_player = self.current_player + target_player_offset
            if target_player > max(self.all_players): target_player -= self.number_of_players

            current_firework_values = self.fireworks

            if (self.other_players_hands_flattened.count(self.player_hands[target_player][self.players_chop_positions[target_player]]) == 1 and self.player_hands[target_player][self.players_chop_positions[target_player]]%10 == 2) and self.player_hands[target_player][self.players_chop_positions[target_player]]%10 > current_firework_values[int(self.player_hands[target_player][self.players_chop_positions[target_player]]/10) - 1]:
                environment_action = (self.player_hand_size)*2 + (self.number_of_players-2+target_player_offset)*self.distribution_length + 2 - 1

        offset_counter += (self.number_of_players-2)
        #6
        if agent_action >= (offset_counter + 6) and agent_action < (6 + offset_counter + self.number_of_players-1):
            target_player_offset = agent_action-offset_counter-6+1
            target_player = self.current_player + target_player_offset
            if target_player > max(self.all_players): target_player -= self.number_of_players

            if self.player_hands[target_player][self.players_chop_positions[target_player]]%10 == 1:
                if self.discard_pile.count(self.player_hands[target_player][self.players_chop_positions[target_player]]) == 2:
                    card_value = self.player_hands[target_player][self.players_chop_positions[target_player]]%10
                    card_colour = int(self.player_hands[target_player][self.players_chop_positions[target_player]]/10)

                    unique_colour = True
                    unique_value = True
                    counter = 0
                    for card in self.player_hands[target_player]:
                        if counter != self.players_chop_positions[target_player]:
                            if card%10 == card_value: unique_value = False
                            if int(card/10) == card_colour: unique_colour = False

                        counter += 1

                    if unique_colour: environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + card_colour - 1
                    elif unique_value: environment_action = (self.player_hand_size)*2 + (self.number_of_players-2+target_player_offset)*self.distribution_length + card_value - 1
                    else: environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + card_colour - 1


            elif self.player_hands[target_player][self.players_chop_positions[target_player]] in self.discard_pile:
                card_value = self.player_hands[target_player][self.players_chop_positions[target_player]]%10
                card_colour = int(self.player_hands[target_player][self.players_chop_positions[target_player]]/10)

                unique_colour = True
                unique_value = True
                counter = 0
                for card in self.player_hands[target_player]:
                    if counter != self.players_chop_positions[target_player]:
                        if card%10 == card_value: unique_value = False
                        if int(card/10) == card_colour: unique_colour = False

                    counter += 1

                if unique_colour: environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + card_colour - 1
                elif unique_value: environment_action = (self.player_hand_size)*2 + (self.number_of_players-2+target_player_offset)*self.distribution_length + card_value - 1
                else: environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + card_colour - 1

        offset_counter += (self.number_of_players-2)
        #7
        if agent_action == (self.convention_total-5):
            target_player_offset = 2  #since we are targeting next next player
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]
            next_next_player = next_player+ 1
            if next_next_player > max(self.all_players): next_next_player = self.all_players[0]

            current_firework_values = self.fireworks

            if len(self.player_hands[next_player]) == self.player_hand_size: card_counter_max = self.player_hand_size - 1
            elif len(self.player_hands[next_player]) == self.player_hand_size-1: card_counter_max = self.player_hand_size - 2  #final round checker since the hands are smaller if deck ran out
            card_counter = 0
            for card in self.player_hands[next_player]:
                if card % 10 == current_firework_values[int(card/10)-1] + 1 and self.players_hinted[next_player][card_counter] != 0: #if next player has a card following on the stack and they know something about the card, we must check to see if next next player has a card following it
                    if self.player_hands[next_next_player].count(card+1) == 1: 
                        colour_to_hint_to_next_next_player = int(card/10)
                        break

                card_counter += 1
                if card_counter > card_counter_max: break

            environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + colour_to_hint_to_next_next_player - 1
                    
        offset_counter += (self.number_of_players-2)
        #8
        if agent_action == (self.convention_total-4):
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]

            hinted_colour = self.previous_actions[-1] - ((self.player_hand_size)*2 + (1)*self.distribution_length) + 1

            current_firework_values = self.fireworks

            if len(self.current_player_hand) == self.player_hand_size: card_counter = self.player_hand_size - 1
            elif len(self.current_player_hand) == self.player_hand_size-1: card_counter = self.player_hand_size - 2  #final round checker since the hands are smaller if deck ran out
            for card in reversed(self.current_player_hinted):   #since we must play newest prompt position
                if int(card/10) == hinted_colour or card%10 == current_firework_values[hinted_colour-1] + 1:
                    environment_action = self.player_hand_size + card_counter
                    break

                card_counter -= 1

        offset_counter += (self.number_of_players-2)
        #9
        if agent_action == (self.convention_total-3):
            target_player_offset = 2  #since we are targeting next next player
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]
            next_next_player = next_player+ 1
            if next_next_player > max(self.all_players): next_next_player = self.all_players[0]

            current_firework_values = self.fireworks

            if len(self.player_hands[next_player]) == self.player_hand_size: card_counter = self.player_hand_size - 1
            elif len(self.player_hands[next_player]) == self.player_hand_size-1: card_counter = self.player_hand_size - 2  #final round checker since the hands are smaller if deck ran out
            for card in reversed(self.player_hands[next_player]):
                if card % 10 == current_firework_values[int(card/10)-1] + 1 and self.players_hinted[next_player][card_counter] == 0: #if next player has a card following on the stack and they know something about the card, we must check to see if next next player has a card following it
                    if self.player_hands[next_next_player].count(card+1) == 1: 
                        colour_to_hint_to_next_next_player = int(card/10)
                        break

                card_counter -= 1

            environment_action = (self.player_hand_size)*2 + (target_player_offset-1)*self.distribution_length + colour_to_hint_to_next_next_player - 1
                    
        offset_counter += (self.number_of_players-2)
        #10
        if agent_action == (self.convention_total-2):
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]

            hinted_colour = self.previous_actions[-1] - ((self.player_hand_size)*2 + (self.number_of_players-2)*self.distribution_length) + 1

            current_firework_values = self.fireworks

            if len(self.current_player_hand) == self.player_hand_size: card_counter = self.player_hand_size - 1
            elif len(self.current_player_hand) == self.player_hand_size-1: card_counter = self.player_hand_size - 2  #final round checker since the hands are smaller if deck ran out
            for card in reversed(self.current_player_hinted):   #since we must play newest finesse position
                if card == 0:
                    environment_action = self.player_hand_size + card_counter
                    break

                card_counter -= 1

        offset_counter += (self.number_of_players-2)
        #11
        if agent_action == (self.convention_total-1): environment_action = self.current_player_chop_position
            
        self.previous_action = environment_action
        self.previous_actions.append(environment_action)
        if len(self.previous_actions) > self.number_of_players-1:self.previous_actions.pop(0)

        return environment_action
    
    def available_conventions(self, env):
        # This function is at the input of the network to extract the current available conventions based on the current observation of a player. 
        # It also incorporates action masking and formatting to allow for the Dopamine agent to interpret it correctly.
        # also uses the confusing (but needed) offset counter

        legal_conventions = []      #positional arguments correlate to actions in description

        self.make_env_usable(env)

        #environment mask
        legal_conventions = legal_conventions + self.env_legal_moves    

        current_firework_values = self.fireworks
        early_game_checker = False
        offset_counter = 0

        #0
        if self.hint_tokens > 0:       
            next_player_in_loop = self.current_player + 1
            for loop_counter in range(self.number_of_players-1):
                if next_player_in_loop > max(self.all_players): next_player_in_loop = self.all_players[0]

                card_counter = 0
                # for card in self.other_player_hand:
                for card in self.player_hands[next_player_in_loop]:
                    ambig_found = False
                    if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                        other_card_counter = 0
                        # for other_card in self.other_player_hand:
                        for other_card in self.player_hands[next_player_in_loop]:
                            if other_card_counter == card_counter:
                                pass

                            elif other_card%10 == card%10 and other_card_counter > card_counter and card_counter != self.players_chop_positions[next_player_in_loop]:
                                ambig_found = True

                            elif other_card%10 == card%10 and other_card_counter == self.players_chop_positions[next_player_in_loop]:
                                ambig_found = True

                            other_card_counter += 1
                        
                        if self.player_hands[next_player_in_loop].count(card) == 1 and not ambig_found:     #we do not hint doubles since it gives useless info
                            legal_conventions.append(0+loop_counter+offset_counter+self.environment_action_space)
                            early_game_checker = True
                            break

                    card_counter += 1

                next_player_in_loop += 1

        offset_counter += (self.number_of_players-2)
        #1
        if self.hint_tokens > 0:
            next_player_in_loop = self.current_player + 1
            for loop_counter in range(self.number_of_players-1):
                if next_player_in_loop > max(self.all_players): next_player_in_loop = self.all_players[0]

                card_counter = 0
                # for card in self.other_player_hand:
                for card in self.player_hands[next_player_in_loop]:
                    ambig_found = False
                    if (card%10)-1 == current_firework_values[int(card/10) - 1]:  #removes colour and decriments to match firework
                        other_card_counter = 0
                        # for other_card in self.other_player_hand:
                        for other_card in self.player_hands[next_player_in_loop]:
                            if other_card_counter == card_counter:
                                pass

                            elif int(other_card/10) == int(card/10) and other_card%10 > card%10 and other_card_counter > card_counter and card_counter != self.players_chop_positions[next_player_in_loop]:
                                ambig_found = True

                            elif int(other_card/10) == int(card/10) and other_card%10 <= current_firework_values[int(card/10)-1] and other_card_counter > card_counter and card_counter != self.players_chop_positions[next_player_in_loop]:
                                ambig_found = True

                            elif int(other_card/10) == int(card/10) and other_card_counter == self.players_chop_positions[next_player_in_loop]:
                                ambig_found = True

                            other_card_counter += 1
                        
                        if self.player_hands[next_player_in_loop].count(card) == 1 and not ambig_found:     #we do not hint doubles since it gives useless info, also avoid ambigious hints
                            legal_conventions.append(1+loop_counter+offset_counter+self.environment_action_space)
                            early_game_checker = True
                            break

                    card_counter += 1

                next_player_in_loop += 1

        offset_counter += (self.number_of_players-2)
        #2
        previous_action_loop_counter = len(self.previous_actions)-1
        for previous_action_loop in range(len(self.previous_actions)):
            if self.previous_actions[previous_action_loop_counter] >= ((self.player_hand_size)*2 + (self.number_of_players-1+previous_action_loop)*self.distribution_length) and self.previous_actions[previous_action_loop_counter] < ((self.player_hand_size)*2 + (self.number_of_players-1+previous_action_loop+1)*self.distribution_length): 
                hinted_value = self.previous_actions[previous_action_loop_counter] - ((self.player_hand_size)*2 + (self.number_of_players-1+previous_action_loop)*self.distribution_length) + 1
                for card in self.current_player_hinted:
                    if (card%10) == hinted_value and (card%10) - 1 in current_firework_values:
                        legal_conventions.append(2+previous_action_loop+offset_counter+self.environment_action_space) #remember, previous action list is inverse from card that was indicated as playable. E.g. [20, 15] 20 is reffering to us and to act on it we must play 5 (not 4)
                        early_game_checker = True
                        break

                    else: pass

            previous_action_loop_counter -= 1

        offset_counter += (self.number_of_players-2)
        #3
        previous_action_loop_counter = len(self.previous_actions)-1
        for previous_action_loop in range(len(self.previous_actions)):
            if self.previous_actions[previous_action_loop_counter] >= ((self.player_hand_size)*2 + (previous_action_loop)*self.distribution_length) and self.previous_actions[previous_action_loop_counter] < ((self.player_hand_size)*2 + (previous_action_loop+1)*self.distribution_length): 
                hinted_colour = self.previous_actions[previous_action_loop_counter] - ((self.player_hand_size)*2 + (previous_action_loop)*self.distribution_length) + 1
                for card in self.current_player_hinted:
                    if int(card/10) == hinted_colour: 
                        legal_conventions.append(3+previous_action_loop+offset_counter+self.environment_action_space)
                        early_game_checker = True
                        break

                    else: pass

            previous_action_loop_counter -= 1
        
        offset_counter += (self.number_of_players-2)
        #4
        if self.hint_tokens > 0:
            next_player_in_loop = self.current_player + 1
            for loop_counter in range(self.number_of_players-1):
                if next_player_in_loop > max(self.all_players): next_player_in_loop = self.all_players[0]

                if self.players_chop_positions[next_player_in_loop] < 5:
                    # if self.other_player_hand[self.other_player_chop_position]%10 == 5:
                    if self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]%10 == 5:
                        legal_conventions.append(4+loop_counter+offset_counter+self.environment_action_space)
                        early_game_checker = True

                next_player_in_loop += 1

        offset_counter += (self.number_of_players-2)
        #5
        if self.hint_tokens > 0:
            next_player_in_loop = self.current_player + 1
            for loop_counter in range(self.number_of_players-1):
                if next_player_in_loop > max(self.all_players): next_player_in_loop = self.all_players[0]

                if self.players_chop_positions[next_player_in_loop] < 5:
                    if (self.other_players_hands_flattened.count(self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]) == 1 and self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]%10 == 2) and self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]%10 > current_firework_values[int(self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]/10) - 1]:
                        legal_conventions.append(5+loop_counter+offset_counter+self.environment_action_space)
                        early_game_checker = True

                next_player_in_loop += 1

        offset_counter += (self.number_of_players-2)
        #6
        if self.hint_tokens > 0:
            next_player_in_loop = self.current_player + 1
            for loop_counter in range(self.number_of_players-1):
                if next_player_in_loop > max(self.all_players): next_player_in_loop = self.all_players[0]

                if self.players_chop_positions[next_player_in_loop] < 5:
                    if self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]%10 == 1:
                        if self.discard_pile.count(self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]) == 2 and self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]%10 > current_firework_values[int(self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]/10) - 1]:
                            legal_conventions.append(6+loop_counter+offset_counter+self.environment_action_space)
                            early_game_checker = True

                    else:
                        if self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]] in self.discard_pile and self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]%10 > current_firework_values[int(self.player_hands[next_player_in_loop][self.players_chop_positions[next_player_in_loop]]/10) - 1]:
                            legal_conventions.append(6+loop_counter+offset_counter+self.environment_action_space)
                            early_game_checker = True

                next_player_in_loop += 1

        offset_counter += (self.number_of_players-2)
        #7                              #prompts must be colour play hints since they must be ambig to the person receiving it, allowing the player in-between to act
        if self.hint_tokens > 0 and self.number_of_players > 2:
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]
            next_next_player = next_player+ 1
            if next_next_player > max(self.all_players): next_next_player = self.all_players[0]

            prompt_found = False

            card_counter = 0
            for card in self.player_hands[next_player]:
                if card % 10 == current_firework_values[int(card/10)-1] + 1 and self.players_hinted[next_player][card_counter] != 0: #if next player has a card following on the stack and they know something about the card, we must check to see if next next player has a card following it
                    #we must first check to make sure this card is not ambig with other card in hand(i.e. it must be the only one or the newest one based on promt conventions)
                    prompt_ambig_found = False
                    if len(self.player_hands[next_player]) == self.player_hand_size: card_check_counter = self.player_hand_size - 1
                    elif len(self.player_hands[next_player]) == self.player_hand_size-1: card_check_counter = self.player_hand_size - 2  #final round checker since the hands are smaller if deck ran out
                    for card_checker in reversed(self.players_hinted[next_player]):
                        if self.player_hands[next_player][card_check_counter] == card: break
                        elif card_checker%10 == card%10 or int(card_checker/10) == int(card/10): prompt_ambig_found = True

                        card_check_counter -= 1

                    if self.players_chop_positions[next_next_player] < 5: #this is needed since there is a difference between that player having a chop and not having a chop
                        if self.player_hands[next_next_player].count(card+1) == 1 and not prompt_ambig_found: #next next player has exactly one card following current card
                            if self.player_hands[next_next_player][self.players_chop_positions[next_next_player]] == card + 1: #card is on chop and can be hinted as play
                                prompt_found = True

                            else:
                                ambig_found = False
                                for next_next_player_card in reversed(self.player_hands[next_next_player]): #we must check that it's the 'newest' card
                                    if next_next_player_card == card + 1:
                                        if not ambig_found: prompt_found = True
                                        break

                                    elif int(next_next_player_card/10) == int(card/10):
                                        ambig_found = True

                    else:
                        if self.player_hands[next_next_player].count(card+1) == 1 and not prompt_ambig_found: #next next player has exactly one card following current card
                            ambig_found = False
                            for next_next_player_card in reversed(self.player_hands[next_next_player]): #we must check that it's the 'newest' card
                                if next_next_player_card == card + 1:
                                    if not ambig_found: prompt_found = True
                                    break

                                elif int(next_next_player_card/10) == int(card/10):
                                    ambig_found = True


                card_counter += 1

            if prompt_found: legal_conventions.append(self.convention_total-5+self.environment_action_space)

        offset_counter += (self.number_of_players-2)
        #8                              #very similar structure to react on colour play hint (3)
        if self.previous_actions[-1] >= ((self.player_hand_size)*2 + (1)*self.distribution_length) and self.previous_actions[-1] < ((self.player_hand_size)*2 + (2)*self.distribution_length) and self.number_of_players > 2:  #checks if last action hinted colour to next next player
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]

            hinted_colour = self.previous_actions[-1] - ((self.player_hand_size)*2 + (1)*self.distribution_length) + 1

            #we must check to see if the hint that was given is 'wrong' and indicates a card that is actually not playable if we do nothing
            ambig_hint_found = False
            for next_player_card in self.player_hands[next_player]:
                if int(next_player_card/10) == hinted_colour and next_player_card%10 == current_firework_values[hinted_colour-1] + 2: 
                    ambig_hint_found = True
                    break

            if ambig_hint_found:
                for card in reversed(self.current_player_hinted):
                    if int(card/10) == hinted_colour or card%10 == current_firework_values[hinted_colour-1] + 1:
                        legal_conventions.append(self.convention_total-4+self.environment_action_space)
                        break

        offset_counter += (self.number_of_players-2)
        #9                              #similar to prompt, it must be a colour hint to cause ambig to target player, and allow in between player to play 'newest' unhinted card
        if self.hint_tokens > 0 and self.number_of_players > 2:
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]
            next_next_player = next_player+ 1
            if next_next_player > max(self.all_players): next_next_player = self.all_players[0]

            finesse_found = False

            if len(self.player_hands[next_player]) == self.player_hand_size: card_counter = self.player_hand_size - 1
            elif len(self.player_hands[next_player]) == self.player_hand_size-1: card_counter = self.player_hand_size - 2  #final round checker since the hands are smaller if deck ran out
            for card in reversed(self.player_hands[next_player]):
                if self.players_hinted[next_player][card_counter] == 0: #finds the newest card in hand (finesse position)
                    if card % 10 == current_firework_values[int(card/10)-1] + 1:
                        if self.player_hands[next_next_player].count(card+1) == 1: #next next player has exactly one card following current card
                            if self.players_chop_positions[next_next_player] < 5: #this is needed since there is a difference between that player having a chop and not having a chop
                                if self.player_hands[next_next_player][self.players_chop_positions[next_next_player]] == card + 1: #card is on chop and can be hinted as play
                                    finesse_found = True

                                else:
                                    ambig_found = False
                                    for next_next_player_card in reversed(self.player_hands[next_next_player]): #we must check that it's the 'newest' card
                                        if next_next_player_card == card + 1:
                                            if not ambig_found: finesse_found = True
                                            break

                                        elif int(next_next_player_card/10) == int(card/10):
                                            ambig_found = True

                            else:
                                ambig_found = False
                                for next_next_player_card in reversed(self.player_hands[next_next_player]): #we must check that it's the 'newest' card
                                    if next_next_player_card == card + 1:
                                        if not ambig_found: finesse_found = True
                                        break

                                    elif int(next_next_player_card/10) == int(card/10):
                                        ambig_found = True

                        else: break #ie next next player does not have card following card in finesse position

                    else: break #ie the card in finesse position does not follow the current stack value

                card_counter -= 1

            if finesse_found: legal_conventions.append(self.convention_total-3+self.environment_action_space)

        offset_counter += (self.number_of_players-2)
        #10
        if self.previous_actions[-1] >= ((self.player_hand_size)*2 + (1)*self.distribution_length) and self.previous_actions[-1] < ((self.player_hand_size)*2 + (2)*self.distribution_length) and self.number_of_players > 2:  #checks if last action hinted colour to next next player
            next_player = self.current_player + 1
            if next_player > max(self.all_players): next_player = self.all_players[0]

            hinted_colour = self.previous_actions[-1] - ((self.player_hand_size)*2 + (1)*self.distribution_length) + 1

            #we must check to see if the hint that was given is 'wrong' and indicates a card that is actually not playable if we do nothing
            ambig_hint_found = False
            for next_player_card in self.player_hands[next_player]:
                if int(next_player_card/10) == hinted_colour and next_player_card%10 == current_firework_values[hinted_colour-1] + 2: 
                    ambig_hint_found = True
                    break

            if ambig_hint_found:
                for card in reversed(self.current_player_hinted):
                    if card == 0:       #ie current player as a card in their finesse position
                        legal_conventions.append(self.convention_total-2+self.environment_action_space)
                        break

        offset_counter += (self.number_of_players-2)
        #11
        # ------------------------------------------------OG--------------------------------------
        if ((not self.discard_pile) and early_game_checker) and self.hint_tokens != 0: #check for early game
            pass
        else:
            if self.hint_tokens < 8:
                if self.current_player_chop_position < 5: legal_conventions.append(self.convention_total-1+self.environment_action_space)
        # ------------------------------------------------------------------------------------------------------------------

        #------------------------------------------Early game removed----------------------------------------------------
        # if self.hint_tokens < 8:
        #     if self.current_player_chop_position < 5: legal_conventions.append(self.convention_total-1+self.environment_action_space)
        #------------------------------------------------------------------------------------------------------------

        return legal_conventions
    
    def make_env_usable(self, env):         #Extracts the needed info from env and translates it into my style of card representation
        # This function extracts all the needed information from the current state and player observation to determine which conventions are available.
        # It receives the entire env as input to extract all the needed game features, but never gives an agent more information than it already
        # has access too. 
        # For all player counts.

        current_player = env.state.cur_player()
        other_players = []
        all_players = []
        for player in range(env.state.num_players()):
            if player != current_player: other_players.append(player)     
            all_players.append(player) 
        self.current_player = current_player
        self.other_players = other_players
        self.all_players = all_players

        self.hint_tokens = env.state.information_tokens()

        fireworks_raw = env.state.fireworks()
        self.fireworks = fireworks_raw

        self.score = env.state.score()

        raw_hands = env.state.player_hands()

        self.player_hands = [None] * env.state.num_players()

        for player in self.all_players:
            player_hand = raw_hands[player]
            self.player_hands[player] = player_hand
            card_counter = 0
            for card in player_hand:
                card_col = card._color
                card_val = card._rank
                self.player_hands[player][card_counter] = (card_col+1)*10 + (card_val+1)
                card_counter += 1

        self.current_player_hand = self.player_hands[current_player]
        self.other_players_hands = self.player_hands.copy()
        self.other_players_hands.pop(current_player)

        self.other_players_hands_flattened = []

        for xx in self.other_players_hands:
            for x in xx: self.other_players_hands_flattened.append(x)

        raw_players_obs = env._make_observation_all_players()
        player_obs = raw_players_obs['player_observations'][current_player]['vectorized']
        legal_moves = raw_players_obs['player_observations'][current_player]['legal_moves_as_int']
        # self.legal_moves = format_legal_moves(legal_moves, env.num_moves())
        self.env_legal_moves = legal_moves

        self.players_hinted = [None] * env.state.num_players()
        for player in self.all_players:
            self.players_hinted[player] = []
            raw_player_obs = raw_players_obs['player_observations'][0]['card_knowledge'][player]
            for hinted_card in raw_player_obs:
                known = 0
                if hinted_card['color'] != None: 
                    if hinted_card['color'] == 'R': known += 10
                    elif hinted_card['color'] == 'Y': known += 20
                    elif hinted_card['color'] == 'G': known += 30
                    elif hinted_card['color'] == 'W': known += 40
                    elif hinted_card['color'] == 'B': known += 50
                if hinted_card['rank'] != None: known += (hinted_card['rank']+1)

                self.players_hinted[player].append(known)

        self.current_player_hinted = self.players_hinted[current_player]
        self.other_players_hinted = self.players_hinted.copy()
        self.other_players_hinted.pop(current_player)

        raw_discard_pile = env.state.discard_pile()
        self.discard_pile = raw_discard_pile
        card_counter = 0
        for card in raw_discard_pile:
            card_col = card._color
            card_val = card._rank
            self.discard_pile[card_counter] = (card_col+1)*10 + (card_val+1)
            card_counter += 1

        self.players_chop_positions = [5] * env.state.num_players()

        player_counter = 0
        for players_hinted_cards in self.players_hinted:
            counter = 0
            found_a_card = False

            for card in players_hinted_cards:
                if card == 0:
                    found_a_card = True
                    break    
                counter += 1

            if found_a_card: self.players_chop_positions[player_counter] = counter
            player_counter += 1

        self.current_player_chop_position = self.players_chop_positions[current_player]
        self.other_players_chop_positions = self.players_chop_positions.copy()
        self.other_players_chop_positions.pop(current_player)

        # pass #for debugging

if __name__ == '__main__':
    # game = rl_env.make(environment_name='Hanabi-Small', num_players=2, pyhanabi_path=None)
    # game = rl_env.make(environment_name='Hanabi-Full', num_players=2, pyhanabi_path=None)
    game = rl_env.make(environment_name='Hanabi-Full', num_players=3, pyhanabi_path=None)
    # convention_encoder = standalone_encoder(game)
    # convention_encoder = simple_official_rules_based_encoder_2p(game)
    convention_encoder = simple_official_rules_based_encoder(game)

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
    





#   Optional extras:
#   7 - if hints == 1 and no card is hintable (ie next on stack in other player's hand), discard oldest card