import streamlit as st
import json
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import numpy as np

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE
    
from classes.description import (
    PlayerDescription, LessonDescription
)

from classes.data_source import Arguments

from classes.embeddings import PlayerEmbeddings,TrolleyEmbeddings,LessonEmbeddings

#from classes.visual import (
    #Visual,DistributionPlot
#)

import utils.sentences as sentences
from utils.gemini import convert_messages_format

openai.api_type = "azure"

class Chat:
    function_names = []
    def __init__(self, chat_state_hash, state="empty"):

        if "chat_state_hash" not in st.session_state or chat_state_hash != st.session_state.chat_state_hash:
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state
    
    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible = True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"What else would you like to know?"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)
                

    def handle_input(self, input):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages. 
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevent information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})
                         
        messages.append({"role": "user", "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```"})

        # Remove all items in messages where content is not a string
        messages = [message for message in messages if isinstance(message["content"], str)]

        # Show the messages in an expander
        st.expander("GPT Messages", expanded=False).write(messages)  

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai
            converted_msgs = convert_messages_format(messages)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"]
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Call the GPT-4 API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=messages
                )
        
            answer=response['choices'][0]['message']['content']
        message = {"role": "assistant", "content": answer}
        
        # Add the returned value to the messages.
        self.messages_to_display.append(message)
   
    def display_content(self,content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try: content.show()
            except: 
                try: st.write(content.get_string())
                except:
                    raise ValueError(f"Message content of type {type(content)} not supported.")


    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns. 
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        #st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message=st.chat_message(name=key, avatar=avatar)   
            with message:
                for message in group:
                    content = message["content"]
                    self.display_content(content)
                

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state

#__________________________________________________________________________________________________________________________________

class LessonChat(Chat):
    def __init__(self, chat_state_hash, overallThesis,arguments, state="empty",gameOver=False):
        self.embeddings = LessonEmbeddings()
        self.arguments =arguments
        self.overallThesis =overallThesis
        self.gameOver=gameOver

        # Initialize the total score as an int and originality score as float
        #self.totalscore = totalscore
        self.originalityscore = np.float64(0.0)
        
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": (
                #"You are talking to a learner about the following topic: " + self.overallThesis + ". "
                "You are an instructor, you guiding the user to learn about  for loops in C programming" 
                
                )
            },
            {"role": "user", "content": (
                f"After these messages you will be interacting with the user who will tell you what they know about classes and objects as used in Object Oriented Programming. "
                f"Your task is to gauge the understading of the user on the topic of classes and objects as use in C++ and then ask them a question that will help fill the knowledge gaps they have. "
                f"You will receive relevant information to answer a user's questions. The relevant information has different options for which is the preffered option. "
                f"You can respond using the number 1 preffered way but if the user responds with the same answer as the previous response, you can number 2 preffered way. "
                f"All user messages will be prefixed with 'user:' and enclosed with ```. "
                f"When responding to the user, speak directly to them. "
                f"When responding to the user query, if there is no relevant information provided, consider the entire conversation with the user and ask them a related question based on the conversation. "
                #f"At the end of the conversation give the user a programming task to practice their knowledge of the for loop in C programming langauge. "
                f"Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                f"At the end of your interaction with the user, provide the user with links to relevant reading material that will enhance their understading of the concepts learnt."
                )
            },
        ]
        return first_messages


    def get_relevant_info(self, query):
 
        #If there is no query then use the last message from the user
        #If there is no query then use the last message from the user
        if query=='':
            query = self.visible_messages[-1]["content"]
       
        numberofarguments = 10
        sidebar_container = st.sidebar.container()

        similaritythreshold = 0.75

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=3)
        results = results.sort_values('similarities', ascending=False)
        sorted_results =results.reset_index(drop=True)
        #st.write(sorted_results)
        #If there are more than one row which gretor than the sililarity threshold, reccomend the most prefered and the second most preffered
        if len(sorted_results)>0:
            greator_similarities=sorted_results['similarities'] >= similaritythreshold
            print(len)
            responce=[greator_similarities]
            for i in range(len(greator_similarities)):
                ret_val = f" This is the number {i+1} preffered way for answering the user question:  " 
                ret_val +="\n".join(sorted_results.loc[[i]]["assistant"])
                responce.append(ret_val)  
                ret_val +="\n".join(sorted_results.loc[[i]]["assistant"].to_list())
            result = " ".join(ret_val)
            return result
            

                    #st.write(ret_val)
            if sorted_results.iloc[0]['similarities'] < similaritythreshold:
                ret_val = "\n\nThe user said:  \n"   
                ret_val +="\n".join(sorted_results["assistant"].to_list())
                ret_val = "Look at the user response, if it is around classes and objects, ask a question that will enhance their undestading of for loops. "
                ret_val += "If the user response is not in the objects and classes tell the user that they should try respond with the relevant topic. "
                    #with sidebar_container:
                            #st.write(f'Novelty: 0/{numberofarguments}')
                            #st.write(f'Total score: {int(np.ceil(self.totalscore))}')
                return ret_val
        if len(results) == 0:
            ret_val="\n\n check the user response, if it is related to classes and objects in general ask the user a question to assess their knowledge on classes and objects as used in Object oriented programming using the C++ programming langauge:\n" 
            ret_val+="If response is not related to object oriented programming in general, tell the user to ask relevant questions"
            with sidebar_container:
                st.write(f'Lesson over! Try again.')
                #st.write(f'Total score: {int(np.ceil(self.totalscore))}')
                self.gameOver=True

                return ret_val
            
            return ret_val

    
    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"Please respond here"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)