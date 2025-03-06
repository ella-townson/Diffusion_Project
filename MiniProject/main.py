from image_generator import ImageGenerator

def main():
    # Create our generator
    generator = ImageGenerator()
    
    while True:
        # Get user input
        prompt = input("\nEnter your image prompt (or 'quit' to exit): ")
        
        if prompt.lower() == 'quit':
            break
            
        try:
            # Generate the image
            filename = generator.generate_image(prompt)
            print(f"Success! Image generated at: {filename}")
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")

if __name__ == "__main__":
    main() 