def main():
    print("AI: Xin chào! Nhập 'exit' để thoát.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("AI: Bye!")
            break

        print(f"Bạn vừa nói: {user_input}")


if __name__ == "__main__":
    main()
