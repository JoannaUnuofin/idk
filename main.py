
from pathlib import Path
from PIL import Image
from huggingface_hub import InferenceClient
from config import HF_API_KEY


MODEL_ID = "black-forest-labs/FLUX.1-dev"


def generate_image_from_text(prompt: str) -> Image.Image:
    """Generate an image from a text prompt."""

    if not HF_API_KEY or HF_API_KEY == "ENTER_YOUR_APIKEY_HERE":
        raise ValueError(
            "Missing API key. Add your Hugging Face token to config.py."
        )

    client = InferenceClient(
        provider="auto",
        api_key=HF_API_KEY,
    )

    image = client.text_to_image(
        prompt=prompt,
        model=MODEL_ID,
    )

    return image


def main():
    print("Welcome to the Text-to-Image Generator!")
    print("Type 'exit' to quit.\n")

    while True:
        prompt = input(
            "Describe the image you want to generate: "
        ).strip()

        if prompt.lower() == "exit":
            print("Goodbye!")
            break

        if not prompt:
            print("Error: Please enter an image description.\n")
            continue

        try:
            print("\nGenerating image. Please wait...\n")

            image = generate_image_from_text(prompt)
            image.show()

            save_option = input(
                "Do you want to save this image? (yes/no): "
            ).strip().lower()

            if save_option == "yes":
                file_name = input(
                    "Enter a filename without an extension: "
                ).strip()

                file_name = "".join(
                    character
                    for character in file_name
                    if character.isalnum()
                    or character in ("_", "-")
                ).strip("_-")

                if not file_name:
                    file_name = "generated_image"

                output_path = Path(f"{file_name}.png")
                image.save(output_path, format="PNG")

                print(
                    f"Image saved to: {output_path.resolve()}\n"
                )

        except Exception as error:
            print(f"An error occurred: {error}\n")

        print("-" * 60)


if __name__ == "__main__":
    main()
