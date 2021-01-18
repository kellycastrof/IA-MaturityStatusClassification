from tkinter import Tk, Frame, Label, PhotoImage, Button, BOTH, X, Y, LEFT, LabelFrame
from tkinter.filedialog import askopenfilename
from predictor import analizar

status={"Class A/":"Banano Verde", 
        "Class B/":"Banano verde-amarillento", 
        "Class C/":"Banano maduro", 
        "Class D/":"Banano sobremaduro",
        "None": "Sin coincidencia"}

def get_root(height, width):
    root = Tk()
    root.title("Maturity Status Classification")
    root.resizable(True, True)
    root.geometry("%dx%d"%(height, width))
    root.config(bg="white")
    root.config(bd=25)
    return root


def analize(name, result):
    image_formats= [("png files", "*.png"),("jpg files", "*.jpg")]
    img_file = askopenfilename(filetypes=image_formats)
    f=img_file.split('/')
    name.config(text=f[-1])
    clase = analizar(img_file)
    textoMB="MobileNet: "+ status[clase]
    result.config(text= textoMB)


def get_body(root):
    title = Label(root, text='Clasificador del estado de madurez del Banano')
    title.pack()
    title.config(fg="black", bg="white", font=("Verdana", 14))
    info = Label(root, text="Esta es una herramienta que te permite conocer el estado de madurez del banano")
    info.pack(pady=10)
    info.config(bg='white')

    source_frame = LabelFrame(root, text="Imagen", width=100, height=80)
    source_frame.pack(fill=X, padx=20, pady=20)
    source_frame.config(bg="white", padx=4, pady=4)

    image_frame = Frame(source_frame)
    image_frame.pack(fill=BOTH, expand=0)
    image_frame.config(bg="white", padx=40, pady=20)

    result_frame = LabelFrame(root, text="Resultado", width=100, height=80)
    result_label = Label(result_frame, text=' ')   

    image_name = Label(image_frame, text=" ", width=40)

    button = Button(image_frame, text="Seleccionar imagen", command=lambda: analize(image_name, result_label))
    button.pack(side=LEFT)
    button.config(foreground='white', bg='gray', relief="raised", borderwidth=0)

    image_name.pack(side=LEFT, padx=4, expand=1, fill=Y)

    result_frame.pack(fill=X, padx=20, pady=20)
    result_frame.config(bg='white')

    result_label.pack(padx=20, pady=20)
    result_label.config(bg='white')


root=get_root(600,500)
get_body(root)

root.mainloop()
