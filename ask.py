from reasoning import answer_question
import argparse

def main():
    parser = argparse.ArgumentParser(description = """ Ask a question "(-q)", 
                                    and to have a response determine how many information be used "(-k)" """)
    
    parser.add_argument("--question","-q", required = True, help = "Question to ask")
    parser.add_argument("--k", "-k", type = int, default = 3 , help = "tok_k chunks to be retreived")
    parser.add_argument("--mode","-m", choices = [v1,v2], default = v1)
    args = parser.parse_args()   
    if args.mode =="v1":               
       answer = answer_question(args.question , args.k)
       print(answer.model_dump_json(indent=2))
    elif args.mode =="v2":
        print("execution of version 2")
     

if __name__ =="__main__":
    main()