import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenrator.utils import read_file,get_table_data
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenrator.mcqgenrator import generate_evaluate_chain
from src.mcqgenrator.logger import logging

with open('C:\Users\vktaw\Desktop\openai\mcqgen\Response.json','r') as f :
    Response_json = json.load(f)

st.title("MCQ's creator application with langchain !!!")

with st.form("user_inputs") :
    uploaded_file = st.file_uploader ("Upload a pdf or a text file only ")
    mcq_count=st.number_input("No. of MCQ ",min_value=3,max_value=50)
    subject= st.text_input("Enter the Subject",max_chars=20)
    tone= st.text_input("Complexity Level Of The Questions ",max_chars=20,placeholder="Simple")
    button = st.form_submit_button("Create MCQ's ")

    if button and uploaded_file is not None and mcq_count and subject and tone :
        with st.spinner("loaing....") :
            try :
                text = read_file(uploaded_file)
                with get_openai_callback as cb :
                    response = generate_evaluate_chain(
                        {
                           "model": "text-davinci-003",  # Use a valid model ID
                           "text": text,
                           "number": mcq_count,
                           "subject": subject,
                           "tone": tone,
                           "response_json": json.dumps(Response_json) 
                        }
                    )
                
            except Exception as e :
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")

            else :
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")    

                if isinstance(response,dict) :
                    quiz= response.get("quiz",None)   
                    if quiz is not None :
                        table_data = get_table_data(quiz)
                        if table_data is not None :
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])

                    else :
                        st.write("Error in the table data")

                else :
                    st.write(response)
                    



