import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import re

# Load environment variables
load_dotenv('api.env')

class DataVizTool:
    def __init__(self, model="gpt-4"):
        self.data = None
        self.model = model
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def load_sample_data(self):
        """Load sample sales data"""
        self.data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'Sales': [1200, 1500, 1300, 1700, 1600, 1800, 
                     2100, 2300, 2500, 2700, 2900, 3100],
            'Expenses': [1000, 1200, 1100, 1300, 1200, 1400, 
                        1600, 1800, 1900, 2000, 2200, 2400]
        })
        return self.data
    
    def load_custom_data(self):
        """Interactive function to load custom data from user input"""
        print("\nEnter your data (type 'done' when finished):")
        print("Format: Month,Sales,Expenses")
        print("Example: Jan,1200,1000")
        
        data_rows = []
        while True:
            row = input("> ")
            if row.lower() == 'done':
                break
            try:
                month, sales, expenses = row.split(',')
                data_rows.append({
                    'Month': month.strip(),
                    'Sales': float(sales.strip()),
                    'Expenses': float(expenses.strip())
                })
            except ValueError:
                print("Invalid format! Please use: Month,Sales,Expenses")
                continue
        
        if data_rows:
            self.data = pd.DataFrame(data_rows)
            return self.data
        else:
            raise ValueError("No data was entered")

    def show_data(self):
        """Display current data"""
        if self.data is not None:
            print("\nCurrent Data:")
            print(self.data)
        else:
            print("No data loaded yet!")

    def extract_code_from_response(self, response_text):
        """Extract only the Python code from the response text"""
        # Remove markdown code blocks if present
        code = response_text.replace('```python', '').replace('```', '')
        
        # Try to find a complete code block
        # Look for patterns that indicate the start of actual code
        code_patterns = [
            r'import.*?matplotlib\.pyplot.*?\n(.*)',
            r'import.*?plotly\.express.*?\n(.*)',
            r'fig,.*?=.*?plt\.subplots.*?\n(.*)',
            r'fig\s*=\s*px\..*?\n(.*)'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, code, re.DOTALL)
            if match:
                code = match.group(0)
                break
        
        # Remove any remaining comments or explanatory text
        lines = code.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and pure comment lines
            if not line or line.startswith('#') and not in_code_block:
                continue
            # Check for code indicators
            if any(indicator in line for indicator in ['import', 'fig', 'plt.', 'px.', 'ax.', '.update_']):
                in_code_block = True
            if in_code_block:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def generate_visualization(self, description):
        """Generate visualization based on description"""
        if self.data is None:
            raise ValueError("Please load data first!")

        # Create prompt for GPT
        prompt = f"""
        Create a Python visualization using this data:
        DataFrame columns: {', '.join(self.data.columns)}
        
        Requirements:
        1. Use matplotlib only
        2. DataFrame variable name is 'data'
        3. Include these exact lines at the start:
           fig, ax = plt.subplots(figsize=(10, 6))
        4. Create visualization for: {description}
        5. Include labels and title
        6. Return only the Python code, no explanations
        
        Example of expected format:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Month'], data['Sales'])
        ax.set_xlabel('Month')
        ax.set_ylabel('Amount')
        ax.set_title('Sales by Month')
        """

        try:
            # Get response from GPT
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5  # Lower temperature for more consistent code
            )
            
            # Extract and clean the code
            viz_code = self.extract_code_from_response(response.choices[0].message.content)
            
            # Create a safe execution environment
            local_vars = {
                'pd': pd,
                'plt': plt,
                'px': px,
                'go': go,
                'data': self.data,
                'fig': None
            }
            
            # Execute the visualization code
            exec(viz_code, globals(), local_vars)
            
            # Return the figure
            if 'fig' in local_vars and local_vars['fig'] is not None:
                return local_vars['fig']
            else:
                return plt.gcf()
                
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            print("\nGenerated code:")
            print(viz_code)
            raise

def main():
    print("Welcome to the Interactive Data Visualization Tool!")
    
    # Initialize the tool
    model_choice = input("\nChoose model (1 for GPT-4, 2 for GPT-3.5-turbo) [1]: ").strip() or "1"
    model = "gpt-4" if model_choice == "1" else "gpt-3.5-turbo"
    
    viz_tool = DataVizTool(model=model)
    
    while True:
        print("\nMenu:")
        print("1. Load sample data")
        print("2. Enter custom data")
        print("3. Show current data")
        print("4. Generate visualization")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        try:
            if choice == '1':
                viz_tool.load_sample_data()
                print("Sample data loaded successfully!")
                viz_tool.show_data()
                
            elif choice == '2':
                viz_tool.load_custom_data()
                print("Custom data loaded successfully!")
                viz_tool.show_data()
                
            elif choice == '3':
                viz_tool.show_data()
                
            elif choice == '4':
                if viz_tool.data is None:
                    print("Please load data first (options 1 or 2)!")
                    continue
                    
                print("\nEnter your visualization request.")
                print("Examples:")
                print("- Create a line plot showing monthly sales trends")
                print("- Make a bar chart comparing sales and expenses")
                print("- Generate a scatter plot of sales vs expenses")
                
                description = input("\nWhat visualization would you like?: ")
                fig = viz_tool.generate_visualization(description)
                plt.show()
                
            elif choice == '5':
                print("\nThank you for using the Data Visualization Tool!")
                break
                
            else:
                print("Invalid choice! Please enter a number between 1 and 5.")
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()
