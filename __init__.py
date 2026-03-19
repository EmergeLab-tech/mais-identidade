bl_info = {
    "name": "+Identidade",
    "author": "+Identidade",
    "version": (3, 0, 1),
    "blender": (5, 0, 0),
    "location": "View3D > N-Panel > +ID",
    "description": "Ferramentas para criação de próteses 3D oncológicas",
    "category": "Medical",
}

import bpy
import os
import json
import subprocess
import tempfile
import shutil
import zipfile
import urllib.request
import numpy as np
from mathutils import Vector, Matrix
from bpy_extras.view3d_utils import region_2d_to_origin_3d, region_2d_to_vector_3d
from bpy.props import (StringProperty, FloatProperty, EnumProperty,
                       BoolProperty, IntProperty, PointerProperty,
                       FloatVectorProperty)
from bpy.types import Panel, Operator, PropertyGroup

# ============================================================
# CAMINHOS
# ============================================================
RAIZ        = r"C:\+Identidade"
OPENMVG_BIN = r"C:\+Identidade\openMVGWin"
OPENMVS_BIN = r"C:\+Identidade\openMVSWin"
EXIFTOOL    = r"C:\+Identidade\ExifTool\exiftool.exe"
SENSOR_DB   = r"C:\+Identidade\openMVGWin\sensor_width_camera_database.txt"
IMAGEMAGICK = r"C:\+Identidade\ImageMagick\mogrify"
COLMAP_EXE  = r"C:\+Identidade\COLMAP\colmap.exe"

BIBLIOTECA_BASE = os.path.join(os.path.expanduser("~"), "Desktop", "+Identidade", "Biblioteca")

# URL de atualização — aponte para onde o zip estiver hospedado
# Exemplo GitHub: https://github.com/SUA_ORG/mais-identidade/releases/latest/download/mais_identidade.zip
UPDATE_VERSION_URL = ""   # URL do version.json remoto
UPDATE_ZIP_URL     = ""   # URL do mais_identidade.zip remoto
_UPDATE_CFG = os.path.join(os.path.dirname(__file__), "update_config.json")

def _carregar_update_cfg():
    if os.path.exists(_UPDATE_CFG):
        with open(_UPDATE_CFG) as f:
            return json.load(f)
    return {"version_url": UPDATE_VERSION_URL, "zip_url": UPDATE_ZIP_URL}

def _salvar_update_cfg(version_url, zip_url):
    with open(_UPDATE_CFG, "w") as f:
        json.dump({"version_url": version_url, "zip_url": zip_url}, f, indent=2)

CATEGORIAS_BIBLIOTECA = [
    ("Nariz",    "Nariz",    ""),
    ("Orelha",   "Orelha",   ""),
    ("Olho",     "Olho",     ""),
    ("Bochecha", "Bochecha", ""),
    ("Mento",    "Mento",    ""),
    ("Outro",    "Outro",    ""),
]

# Landmarks anatômicos faciais padrão
# (código, nome, descrição, é linha_media)
LANDMARKS = [
    ("Gl",  "Glabela",             "Ponto mais projetado da testa",           True),
    ("N",   "Násio",               "Raiz do nariz (junção testa-nariz)",       True),
    ("Prn", "Ponta do Nariz",      "Ponta mais anterior do nariz",            True),
    ("Sn",  "Subnasal",            "Base do septo nasal",                     True),
    ("Ls",  "Lábio Superior",      "Borda superior do lábio",                 True),
    ("Li",  "Lábio Inferior",      "Borda inferior do lábio",                 True),
    ("Pog", "Pogônio",             "Ponto mais anterior do mento",            True),
    ("Me",  "Mênton",              "Ponto mais inferior do mento",            True),
    ("EnR", "Canto Medial Olho D", "Canto interno do olho direito",           False),
    ("EnL", "Canto Medial Olho E", "Canto interno do olho esquerdo",          False),
    ("ExR", "Canto Lateral Olho D","Canto externo do olho direito",           False),
    ("ExL", "Canto Lateral Olho E","Canto externo do olho esquerdo",          False),
    ("TR",  "Tragus Direito",      "Referência da orelha direita",            False),
    ("TL",  "Tragus Esquerdo",     "Referência da orelha esquerda",           False),
    ("ChR", "Comissura Labial D",  "Canto da boca direito",                   False),
    ("ChL", "Comissura Labial E",  "Canto da boca esquerdo",                  False),
]
LANDMARK_CODES = [lm[0] for lm in LANDMARKS]

# ============================================================
# PROPRIEDADES
# ============================================================
class PlusIDProperties(PropertyGroup):

    nome_paciente:      StringProperty(name="Nome",      default="")
    sobrenome_paciente: StringProperty(name="Sobrenome", default="")

    path_photo: StringProperty(name="Fotos", subtype='DIR_PATH', default="")

    motor_foto: EnumProperty(
        name="Motor",
        items=[
            ('OPENMVG', "OpenMVG + OpenMVS", "Pipeline padrão, rápido e confiável"),
            ('COLMAP',  "COLMAP + OpenMVS",  "Mais preciso em superfícies complexas"),
        ],
        default='OPENMVG'
    )
    qualidade_foto: EnumProperty(
        name="Qualidade",
        items=[
            ('RAPIDO', "Rápido",  "Menor detalhe, processa mais rápido"),
            ('NORMAL', "Normal",  "Equilíbrio entre detalhe e tempo"),
            ('ALTO',   "Alto",    "Máximo detalhe, processa mais lento"),
        ],
        default='NORMAL'
    )
    usar_gpu: BoolProperty(
        name="Usar GPU (CUDA)",
        description="Acelera o COLMAP com placa NVIDIA. Requer CUDA instalado.",
        default=False
    )

    medida_real: FloatProperty(
        name="Distância Real (mm)", default=100.0, min=1.0
    )

    eixo_espelho: EnumProperty(
        name="Eixo",
        items=[
            ('X', 'X', 'Esquerda / Direita'),
            ('Y', 'Y', 'Frente / Trás'),
            ('Z', 'Z', 'Cima / Baixo'),
        ],
        default='X'
    )

    cat_biblioteca:  EnumProperty(name="Categoria", items=CATEGORIAS_BIBLIOTECA)
    nome_biblioteca: StringProperty(name="Nome da Peça", default="")

    prof_trabalho: FloatProperty(name="Profundidade (mm)", default=20.0, min=1.0)
    esp_parede:    FloatProperty(name="Espessura da Parede (mm)", default=3.0, min=1.0)
    oco_trabalho:  BoolProperty(name="Deixar Oco", default=True)

    # Verificação de impressão
    chk_manifold:   BoolProperty(name="Geometria não-manifold",  default=True)
    chk_normais:    BoolProperty(name="Normais invertidas",       default=True)
    chk_espessura:  BoolProperty(name="Espessura mínima",         default=True)
    chk_volume:     BoolProperty(name="Volume (peça fechada)",    default=True)
    chk_dimensoes:  BoolProperty(name="Dimensões da impressora",  default=False)
    espessura_min:  FloatProperty(name="Espessura mínima (mm)",   default=1.2, min=0.1)
    dim_impressora: FloatVectorProperty(
        name="Tam. máx. impressora (mm)", size=3,
        default=(220.0, 220.0, 250.0), min=1.0
    )
    autofix_manifold: BoolProperty(name="Auto-corrigir manifold", default=True)
    autofix_normais:  BoolProperty(name="Auto-corrigir normais",  default=True)

    # Atualização
    update_version_url: StringProperty(
        name="URL version.json",
        description="URL remota do arquivo version.json",
        default=""
    )
    update_zip_url: StringProperty(
        name="URL do ZIP",
        description="URL remota do mais_identidade.zip",
        default=""
    )


# ============================================================
# UTILITÁRIOS
# ============================================================
def pasta_paciente(context):
    return context.scene.get("pasta_paciente",
           os.path.join(os.path.expanduser("~"), "Desktop"))


def garantir_camera_no_banco(cam_model):
    if not os.path.exists(SENSOR_DB):
        return
    with open(SENSOR_DB, 'r') as f:
        content = f.read()
    if cam_model + ";" not in content:
        with open(SENSOR_DB, 'a') as f:
            f.write(f"\n{cam_model}; 3.80")


def get_camera_model(photo_path):
    files = [f for f in os.listdir(photo_path)
             if f.lower().endswith(('.jpg', '.jpeg', '.heic'))]
    if not files:
        return None
    try:
        r = subprocess.run(
            [EXIFTOOL, '-Model', '-s3', os.path.join(photo_path, files[0])],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def converter_heic(photo_path, tmpdir):
    if not any(f.lower().endswith('.heic') for f in os.listdir(photo_path)):
        return photo_path
    jpg_dir = os.path.join(tmpdir, "JPG")
    os.makedirs(jpg_dir, exist_ok=True)
    subprocess.call(
        f'cd /d "{photo_path}" && "{IMAGEMAGICK}" -format jpg *.HEIC && move *.jpg "{jpg_dir}"',
        shell=True
    )
    return jpg_dir


def pos_importacao(context, obj, nome):
    obj.name = nome
    bpy.ops.object.shade_smooth()
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN')

    mod_sub = obj.modifiers.new("Subdivisão", type='SUBSURF')
    mod_sub.levels = 1
    mod_sub.render_levels = 2
    mod_sub.show_viewport = False

    tex_img = next((img for img in bpy.data.images
                    if "map_Kd" in img.name or "texture" in img.name.lower()), None)
    if tex_img:
        tex = bpy.data.textures.new("Displacement_Foto", type='IMAGE')
        tex.image = tex_img
        mod_disp = obj.modifiers.new("Displacement", type='DISPLACE')
        mod_disp.texture        = tex
        mod_disp.texture_coords = 'UV'
        mod_disp.strength       = 3.2
        mod_disp.mid_level      = 0.5
        mod_disp.show_viewport  = False

    for mod in obj.modifiers:
        mod.show_expanded = False

    for area in context.screen.areas:
        if area.type == 'VIEW_3D':
            with context.temp_override(area=area):
                bpy.ops.view3d.view_all()
            break


def landmark_empty_name(code):
    return f"LM_{code}"


def get_landmark_pos(code):
    obj = bpy.data.objects.get(landmark_empty_name(code))
    if obj:
        return list(obj.location)
    return None


def midline_landmarks():
    """Retorna posições dos landmarks de linha média que estão na cena."""
    pts = []
    for code, _, _, is_midline in LANDMARKS:
        if is_midline:
            pos = get_landmark_pos(code)
            if pos:
                pts.append(pos)
    return pts


# ============================================================
# OPERADORES - PACIENTE
# ============================================================
class PLUSID_OT_SalvarPaciente(Operator):
    bl_idname = "plusid.salvar_paciente"
    bl_label = "Salvar Paciente"

    def execute(self, context):
        props = context.scene.plusid
        nome = f"{props.sobrenome_paciente}_{props.nome_paciente}".strip("_").replace(" ", "_")
        if not nome:
            self.report({'ERROR'}, "Digite o nome do paciente!")
            return {'CANCELLED'}
        pasta = os.path.join(os.path.expanduser("~"), "Desktop", nome)
        os.makedirs(pasta, exist_ok=True)
        context.scene["pasta_paciente"] = pasta
        self.report({'INFO'}, f"Pasta criada: {pasta}")
        return {'FINISHED'}


# ============================================================
# OPERADORES - FOTOGRAMETRIA
# ============================================================
class PLUSID_OT_Fotogrametria(Operator):
    bl_idname = "plusid.fotogrametria"
    bl_label = "Iniciar Fotogrametria"

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _run(self, cmd_list, titulo=""):
        """Abre janela CMD visível, executa o comando e aguarda o término."""
        titulo   = titulo or os.path.basename(cmd_list[0])
        cmd_str  = subprocess.list2cmdline(cmd_list)
        # title muda o nome da janela; /c fecha ao terminar
        wrapped  = f'title +Identidade  [{titulo}] && {cmd_str}'
        proc = subprocess.Popen(
            ['cmd', '/c', wrapped],
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )
        return proc.wait()

    def _prog(self, wm, step, total, label=""):
        """Atualiza a barra de progresso do Blender e imprime no console."""
        wm.progress_update(step)
        pct = int(step / total * 100)
        print(f"=== [{pct:3d}%]  {label}  ({step}/{total}) ===")
        try:
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Pipeline OpenMVG + OpenMVS  (9 etapas)
    # ----------------------------------------------------------------
    def _openmvg_pipeline(self, photo_path, tmpdir, qualidade, wm):
        d_factor = {'RAPIDO': '8',  'NORMAL': '16', 'ALTO': '32'}[qualidade]
        smooth   = {'RAPIDO': '4',  'NORMAL': '2',  'ALTO': '1' }[qualidade]

        openmvg_dir = os.path.join(tmpdir, "OpenMVG")
        matches_dir = os.path.join(openmvg_dir, "matches")
        recon_dir   = os.path.join(openmvg_dir, "reconstruction_sequential")
        mvs_dir     = os.path.join(tmpdir, "MVS")
        for d in [matches_dir, recon_dir, mvs_dir]:
            os.makedirs(d, exist_ok=True)

        bin_  = OPENMVG_BIN
        TOTAL = 9

        self._prog(wm, 1, TOTAL, "Image Listing")
        self._run([
            os.path.join(bin_, "openMVG_main_SfMInit_ImageListing"),
            "-i", photo_path, "-o", matches_dir, "-d", SENSOR_DB,
        ], "1/9 Image Listing")

        sfm_json = os.path.join(matches_dir, "sfm_data.json")
        if not os.path.exists(sfm_json):
            print("ERRO: sfm_data.json não gerado. Verifique fotos e banco de câmeras.")
            return None

        self._prog(wm, 2, TOTAL, "Compute Features")
        self._run([
            os.path.join(bin_, "openMVG_main_ComputeFeatures"),
            "-i", sfm_json, "-o", matches_dir, "-m", "SIFT",
        ], "2/9 Compute Features")

        self._prog(wm, 3, TOTAL, "Compute Matches")
        self._run([
            os.path.join(bin_, "openMVG_main_ComputeMatches"),
            "-i", sfm_json, "-o", matches_dir,
        ], "3/9 Compute Matches")

        self._prog(wm, 4, TOTAL, "Incremental SfM")
        self._run([
            os.path.join(bin_, "openMVG_main_IncrementalSfM"),
            "-i", sfm_json, "-m", matches_dir, "-o", recon_dir,
        ], "4/9 Incremental SfM")

        sfm_data = os.path.join(recon_dir, "sfm_data.bin")
        if not os.path.exists(sfm_data):
            print("ERRO: SfM falhou. Fotos insuficientes ou sem sobreposição.")
            return None

        self._prog(wm, 5, TOTAL, "Colorize SfM")
        self._run([
            os.path.join(bin_, "openMVG_main_ComputeSfM_DataColor"),
            "-i", sfm_data,
            "-o", os.path.join(recon_dir, "colorized.ply"),
        ], "5/9 Colorize")

        self._prog(wm, 6, TOTAL, "OpenMVG → OpenMVS")
        os.chdir(tmpdir)
        mvs_scene = os.path.join(mvs_dir, "scene.mvs")
        self._run([
            os.path.join(bin_, "openMVG_main_openMVG2openMVS"),
            "-i", sfm_data, "-o", mvs_scene,
        ], "6/9 MVG→MVS")

        self._prog(wm, 7, TOTAL, "Densify Point Cloud")
        self._run([
            os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
            "--estimate-normals", "1", mvs_scene,
        ], "7/9 Densify")

        self._prog(wm, 8, TOTAL, "Reconstruct Mesh")
        self._run([
            os.path.join(OPENMVS_BIN, "ReconstructMesh"),
            "-d", d_factor, "--smooth", smooth,
            os.path.join(mvs_dir, "scene_dense.mvs"),
        ], "8/9 Reconstruct Mesh")

        self._prog(wm, 9, TOTAL, "Texture Mesh")
        self._run([
            os.path.join(OPENMVS_BIN, "TextureMesh"),
            "--export-type", "obj",
            os.path.join(mvs_dir, "scene_dense_mesh.mvs"),
        ], "9/9 Texture")

        obj_file = os.path.join(mvs_dir, "scene_dense_mesh_texture.obj")
        return obj_file if os.path.exists(obj_file) else None

    # ----------------------------------------------------------------
    # Pipeline COLMAP + OpenMVS  (7 etapas)
    # ----------------------------------------------------------------
    def _colmap_pipeline(self, photo_path, tmpdir, qualidade, usar_gpu, wm):
        max_size = {'RAPIDO': '1500', 'NORMAL': '2400', 'ALTO': '3200'}[qualidade]
        geom     = {'RAPIDO': '0',    'NORMAL': '0',    'ALTO': '1'   }[qualidade]
        gpu_val  = '1' if usar_gpu else '0'

        db     = os.path.join(tmpdir, "database.db")
        sparse = os.path.join(tmpdir, "sparse")
        dense  = os.path.join(tmpdir, "dense")
        os.makedirs(sparse, exist_ok=True)
        os.makedirs(dense,  exist_ok=True)

        TOTAL = 7

        self._prog(wm, 1, TOTAL, "Feature Extraction")
        self._run([COLMAP_EXE, "feature_extractor",
                   "--database_path", db,
                   "--image_path",    photo_path,
                   "--SiftExtraction.max_image_size", max_size,
                   "--SiftExtraction.use_gpu", gpu_val],
                  "1/7 Feature Extraction")

        self._prog(wm, 2, TOTAL, "Exhaustive Matching")
        self._run([COLMAP_EXE, "exhaustive_matcher",
                   "--database_path", db,
                   "--SiftMatching.use_gpu", gpu_val],
                  "2/7 Exhaustive Matching")

        self._prog(wm, 3, TOTAL, "Sparse Reconstruction")
        self._run([COLMAP_EXE, "mapper",
                   "--database_path", db,
                   "--image_path",    photo_path,
                   "--output_path",   sparse],
                  "3/7 Sparse SfM")

        sparse0 = os.path.join(sparse, "0")
        if not os.path.exists(sparse0):
            return None

        self._prog(wm, 4, TOTAL, "Image Undistortion")
        self._run([COLMAP_EXE, "image_undistorter",
                   "--image_path",  photo_path,
                   "--input_path",  sparse0,
                   "--output_path", dense,
                   "--output_type", "COLMAP"],
                  "4/7 Undistort")

        self._prog(wm, 5, TOTAL, "Dense Stereo Matching")
        self._run([COLMAP_EXE, "patch_match_stereo",
                   "--workspace_path",                    dense,
                   "--PatchMatchStereo.geom_consistency", geom],
                  "5/7 Dense MVS")

        self._prog(wm, 6, TOTAL, "Stereo Fusion")
        fused = os.path.join(dense, "fused.ply")
        self._run([COLMAP_EXE, "stereo_fusion",
                   "--workspace_path", dense,
                   "--output_path",    fused],
                  "6/7 Fusion")

        self._prog(wm, 7, TOTAL, "Poisson Meshing")
        mesh = os.path.join(dense, "mesh.ply")
        self._run([COLMAP_EXE, "poisson_mesher",
                   "--input_path",  fused,
                   "--output_path", mesh],
                  "7/7 Mesh")

        return mesh if os.path.exists(mesh) else None

    # ----------------------------------------------------------------
    # Execute
    # ----------------------------------------------------------------
    def execute(self, context):
        props      = context.scene.plusid
        photo_path = bpy.path.abspath(props.path_photo).rstrip("/\\")

        if not photo_path or not os.path.isdir(photo_path):
            self.report({'ERROR'}, "Selecione uma pasta de fotos válida!")
            return {'CANCELLED'}

        TOTAL = 9 if props.motor_foto == 'OPENMVG' else 7
        wm    = context.window_manager
        wm.progress_begin(0, TOTAL)

        tmpdir = tempfile.mkdtemp()
        print(f"\n=== +IDENTIDADE | {props.motor_foto} / {props.qualidade_foto} ===")

        try:
            photo_path = converter_heic(photo_path, tmpdir)

            cam = get_camera_model(photo_path)
            if cam:
                subprocess.call([EXIFTOOL, '-all=', photo_path])
                subprocess.call([EXIFTOOL, '-overwrite_original',
                                  f'-Model={cam}', '-FocalLength=4', photo_path])
                garantir_camera_no_banco(cam)

            self.report({'INFO'}, f"Processando ({props.motor_foto})… acompanhe pela janela CMD.")

            if props.motor_foto == 'OPENMVG':
                resultado = self._openmvg_pipeline(photo_path, tmpdir, props.qualidade_foto, wm)
            else:
                resultado = self._colmap_pipeline(photo_path, tmpdir,
                                                   props.qualidade_foto, props.usar_gpu, wm)

            if not resultado:
                self.report({'ERROR'}, "Fotogrametria falhou. Veja a janela CMD.")
                wm.progress_end()
                return {'CANCELLED'}

            ext = os.path.splitext(resultado)[1].lower()
            if ext == '.obj':
                bpy.ops.wm.obj_import(filepath=resultado)
            else:
                bpy.ops.wm.ply_import(filepath=resultado)

            obj = context.active_object
            if obj:
                pos_importacao(context, obj, "PACIENTE")

            self.report({'INFO'}, "Fotogrametria concluída!")

        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Erro: {e}")
            wm.progress_end()
            return {'CANCELLED'}

        wm.progress_end()
        return {'FINISHED'}


# ============================================================
# OPERADORES - SNAP À SUPERFÍCIE
# ============================================================
class PLUSID_OT_SnapPonto(Operator):
    bl_idname = "plusid.snap_ponto"
    bl_label = "Clicar na Superfície"
    bl_description = "Clique diretamente na malha para gravar o ponto exato"

    modo: StringProperty(default="escala_a")

    def _label(self):
        if self.modo == "escala_a": return "Ponto A (escala)"
        if self.modo == "escala_b": return "Ponto B (escala)"
        if self.modo.startswith("landmark_"):
            code = self.modo.split("_", 1)[1]
            entry = next((lm for lm in LANDMARKS if lm[0] == code), None)
            return entry[1] if entry else code
        parts = self.modo.split("_")
        label = "Escaneamento" if parts[1] == "scan" else "Fotogrametria"
        return f"Ponto {parts[2]} ({label})"

    def _store(self, context, location):
        pos = [location.x, location.y, location.z]
        if self.modo == "escala_a":
            context.scene["escala_ponto_a"] = pos
        elif self.modo == "escala_b":
            context.scene["escala_ponto_b"] = pos
        elif self.modo.startswith("landmark_"):
            code = self.modo.split("_", 1)[1]
            # Cria ou move o Empty do landmark
            empty_name = landmark_empty_name(code)
            empty = bpy.data.objects.get(empty_name)
            if not empty:
                bpy.ops.object.empty_add(type='SPHERE', radius=0.005)
                empty = context.active_object
                empty.name = empty_name
                # Cor por coleção
                entry = next((lm for lm in LANDMARKS if lm[0] == code), None)
                if entry and entry[3]:  # linha média = amarelo
                    empty.color = (1.0, 0.9, 0.0, 1.0)
                else:
                    empty.color = (0.0, 0.7, 1.0, 1.0)
            empty.location = location
        else:
            parts = self.modo.split("_")
            context.scene[f"align_{parts[1]}_p{parts[2]}"] = pos

    def modal(self, context, event):
        context.area.tag_redraw()
        if event.type == 'MOUSEMOVE':
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            coord     = (event.mouse_region_x, event.mouse_region_y)
            origin    = region_2d_to_origin_3d(self.region, self.rv3d, coord)
            direction = region_2d_to_vector_3d(self.region, self.rv3d, coord)
            hit, loc, *_ = context.scene.ray_cast(context.view_layer, origin, direction)
            if hit:
                context.scene.cursor.location = loc
                self._store(context, loc)
                self.report({'INFO'}, f"{self._label()} gravado.")
            else:
                self.report({'WARNING'}, "Nenhuma superfície encontrada.")
            return {'FINISHED'}
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Use no Viewport 3D.")
            return {'CANCELLED'}
        self.region = context.region
        self.rv3d   = context.region_data
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, f"Clique na superfície: {self._label()}")
        return {'RUNNING_MODAL'}


# ============================================================
# OPERADORES - ESCALA
# ============================================================
class PLUSID_OT_AplicarEscala(Operator):
    bl_idname = "plusid.aplicar_escala"
    bl_label = "Aplicar Escala"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.plusid
        pt_a  = context.scene.get("escala_ponto_a")
        pt_b  = context.scene.get("escala_ponto_b")

        if not pt_a or not pt_b:
            self.report({'ERROR'}, "Grave os Pontos A e B primeiro!")
            return {'CANCELLED'}

        dist_digital = (Vector(pt_b) - Vector(pt_a)).length
        if dist_digital < 0.0001:
            self.report({'ERROR'}, "Pontos muito próximos!")
            return {'CANCELLED'}

        fator = (props.medida_real / 1000.0) / dist_digital
        obj   = context.active_object
        if not obj:
            self.report({'ERROR'}, "Selecione o objeto!")
            return {'CANCELLED'}

        obj.scale = tuple(s * fator for s in obj.scale)
        bpy.ops.object.transform_apply(scale=True)
        self.report({'INFO'},
            f"Escala aplicada! Fator {fator:.4f} | "
            f"{dist_digital*1000:.2f} mm → {props.medida_real:.1f} mm real")
        return {'FINISHED'}


# ============================================================
# OPERADORES - RECORTE
# ============================================================
class PLUSID_OT_DesenharLinha(Operator):
    bl_idname = "plusid.desenhar_linha"
    bl_label = "Desenhar Linha de Corte"
    def execute(self, context):
        bpy.ops.gpencil.annotate('INVOKE_DEFAULT', mode='DRAW_POLY')
        return {'FINISHED'}

class PLUSID_OT_CortarDesenho(Operator):
    bl_idname = "plusid.cortar_desenho"
    bl_label = "Cortar pelo Desenho"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecione uma malha!")
            return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        try:
            bpy.ops.mesh.knife_project(cut_through=True)
            bpy.ops.mesh.select_all(action='INVERT')
            bpy.ops.mesh.delete(type='FACE')
        except Exception as e:
            self.report({'WARNING'}, f"Use Knife Project manualmente. {e}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


# ============================================================
# OPERADORES - ESPELHAMENTO
# ============================================================
class PLUSID_OT_EspelharProtese(Operator):
    bl_idname = "plusid.espelhar_protese"
    bl_label = "Copiar e Espelhar"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "Selecione o objeto!")
            return {'CANCELLED'}
        obj.name = "PACIENTE"
        bpy.ops.object.duplicate()
        novo = context.active_object
        novo.name = "PROTESE"
        eixo = context.scene.plusid.eixo_espelho
        idx  = {'X': 0, 'Y': 1, 'Z': 2}[eixo]
        s = list(novo.scale); s[idx] *= -1; novo.scale = tuple(s)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, f"Prótese criada (espelho {eixo}).")
        return {'FINISHED'}


# ============================================================
# OPERADORES - ESCULTURA
# ============================================================
class PLUSID_OT_EsculturaGrab(Operator):
    bl_idname = "plusid.escultura_grab"; bl_label = "Grab"
    def execute(self, context):
        bpy.ops.object.mode_set(mode='SCULPT')
        bpy.ops.wm.tool_set_by_id(name="builtin_brush.Grab")
        return {'FINISHED'}

class PLUSID_OT_EsculturaSmooth(Operator):
    bl_idname = "plusid.escultura_smooth"; bl_label = "Smooth"
    def execute(self, context):
        bpy.ops.object.mode_set(mode='SCULPT')
        bpy.ops.wm.tool_set_by_id(name="builtin_brush.Smooth")
        return {'FINISHED'}

class PLUSID_OT_EsculturaClay(Operator):
    bl_idname = "plusid.escultura_clay"; bl_label = "Clay Strips"
    def execute(self, context):
        bpy.ops.object.mode_set(mode='SCULPT')
        bpy.ops.wm.tool_set_by_id(name="builtin_brush.Clay Strips")
        return {'FINISHED'}

class PLUSID_OT_ModoObjeto(Operator):
    bl_idname = "plusid.modo_objeto"; bl_label = "Voltar ao Modo Objeto"
    def execute(self, context):
        bpy.ops.object.mode_set(mode='OBJECT')
        return {'FINISHED'}


# ============================================================
# OPERADORES - BOOLEANO
# ============================================================
class PLUSID_OT_BooleanaEncaixe(Operator):
    bl_idname = "plusid.booleana_encaixe"
    bl_label = "Gerar Encaixe (Diferença)"
    bl_description = "Ativo = PROTESE, Selecionado = PACIENTE"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        outros = [o for o in context.selected_objects if o != context.active_object]
        if not outros:
            self.report({'ERROR'}, "Selecione também o objeto cortador!"); return {'CANCELLED'}
        protese = context.active_object; cortador = outros[0]
        mod = protese.modifiers.new("Encaixe", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'; mod.object = cortador; mod.solver = 'FAST'
        bpy.ops.object.modifier_apply(modifier="Encaixe")
        cortador.hide_set(True)
        self.report({'INFO'}, "Encaixe gerado!"); return {'FINISHED'}

class PLUSID_OT_BooleanaUniao(Operator):
    bl_idname = "plusid.booleana_uniao"; bl_label = "União"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        outros = [o for o in context.selected_objects if o != context.active_object]
        if not outros:
            self.report({'ERROR'}, "Selecione dois objetos!"); return {'CANCELLED'}
        base = context.active_object; outro = outros[0]
        mod = base.modifiers.new("Uniao", type='BOOLEAN')
        mod.operation = 'UNION'; mod.object = outro; mod.solver = 'FAST'
        bpy.ops.object.modifier_apply(modifier="Uniao")
        bpy.data.objects.remove(outro, do_unlink=True)
        self.report({'INFO'}, "União aplicada!"); return {'FINISHED'}


# ============================================================
# OPERADORES - BIBLIOTECA
# ============================================================
def _caminho_biblioteca(categoria, nome=None):
    pasta = os.path.join(BIBLIOTECA_BASE, categoria)
    os.makedirs(pasta, exist_ok=True)
    return os.path.join(pasta, nome + ".obj") if nome else pasta

def _listar_biblioteca(categoria):
    pasta = _caminho_biblioteca(categoria)
    return [f[:-4] for f in os.listdir(pasta) if f.endswith(".obj")] if os.path.isdir(pasta) else []

class PLUSID_OT_SalvarBiblioteca(Operator):
    bl_idname = "plusid.salvar_biblioteca"; bl_label = "Salvar na Biblioteca"
    def execute(self, context):
        props = context.scene.plusid
        nome  = props.nome_biblioteca.strip().replace(" ", "_")
        if not nome:
            self.report({'ERROR'}, "Digite um nome!"); return {'CANCELLED'}
        obj = context.active_object
        if not obj:
            self.report({'ERROR'}, "Selecione o objeto!"); return {'CANCELLED'}
        filepath = _caminho_biblioteca(props.cat_biblioteca, nome)
        bpy.ops.wm.obj_export(filepath=filepath, export_selected_objects=True, export_materials=True)
        with open(filepath.replace(".obj", ".json"), "w") as f:
            json.dump({"nome": nome, "categoria": props.cat_biblioteca}, f, ensure_ascii=False, indent=2)
        self.report({'INFO'}, f"Salvo: {filepath}"); return {'FINISHED'}

class PLUSID_OT_CarregarBiblioteca(Operator):
    bl_idname = "plusid.carregar_biblioteca"; bl_label = "Carregar"
    nome_peca: StringProperty(default="")
    def execute(self, context):
        props    = context.scene.plusid
        nome     = self.nome_peca or props.nome_biblioteca.strip()
        filepath = _caminho_biblioteca(props.cat_biblioteca, nome)
        if not os.path.exists(filepath):
            self.report({'ERROR'}, f"Não encontrado: {filepath}"); return {'CANCELLED'}
        bpy.ops.wm.obj_import(filepath=filepath)
        obj = context.active_object
        if obj: obj.name = f"BIBLIO_{nome}"
        self.report({'INFO'}, f"Carregado: {nome}"); return {'FINISHED'}


# ============================================================
# OPERADORES - MODELO DE TRABALHO
# ============================================================
class PLUSID_OT_CriarModeloTrabalho(Operator):
    bl_idname = "plusid.criar_modelo_trabalho"
    bl_label = "Criar Modelo de Trabalho"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.plusid
        obj   = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecione a malha!"); return {'CANCELLED'}

        bpy.ops.object.duplicate()
        bloco = context.active_object
        bloco.name = "MODELO_TRABALHO"
        prof_m = props.prof_trabalho / 1000.0

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.extrude_region_move(
            MESH_OT_extrude_region={"use_normal_flip": False, "mirror": False},
            TRANSFORM_OT_translate={"value": (0, 0, -prof_m),
                                    "orient_type": 'NORMAL',
                                    "constraint_axis": (False, False, True)}
        )
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold(extend=False, use_wire=False,
                                         use_boundary=True, use_multi_face=False,
                                         use_non_contiguous=False, use_verts=False)
        bpy.ops.mesh.edge_face_add()
        bpy.ops.object.mode_set(mode='OBJECT')

        if props.oco_trabalho:
            mod = bloco.modifiers.new("Parede", type='SOLIDIFY')
            mod.thickness     = -(props.esp_parede / 1000.0)
            mod.offset        = 1.0
            mod.use_rim       = True
            mod.use_rim_only  = False
            bpy.ops.object.modifier_apply(modifier="Parede")

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, "Modelo de trabalho criado!"); return {'FINISHED'}


# ============================================================
# OPERADORES - ESCANEAMENTO E ALINHAMENTO
# ============================================================
class PLUSID_OT_ImportarEscaneamento(Operator):
    bl_idname = "plusid.importar_escaneamento"; bl_label = "Importar Escaneamento"
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.stl;*.obj;*.ply", options={'HIDDEN'})
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self); return {'RUNNING_MODAL'}
    def execute(self, context):
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext == '.stl':   bpy.ops.wm.stl_import(filepath=self.filepath)
        elif ext == '.obj': bpy.ops.wm.obj_import(filepath=self.filepath)
        elif ext == '.ply': bpy.ops.wm.ply_import(filepath=self.filepath)
        else:
            self.report({'ERROR'}, "Use STL, OBJ ou PLY."); return {'CANCELLED'}
        obj = context.active_object
        if obj:
            obj.name = "ESCANEAMENTO"
            if not obj.data.materials:
                mat = bpy.data.materials.new("Mat_Escaneamento")
                mat.diffuse_color = (0.2, 0.6, 1.0, 0.8)
                obj.data.materials.append(mat)
        self.report({'INFO'}, f"Importado: {os.path.basename(self.filepath)}"); return {'FINISHED'}

class PLUSID_OT_AlinharEscaneamento(Operator):
    bl_idname = "plusid.alinhar_escaneamento"; bl_label = "Alinhar Escaneamento"
    bl_options = {'REGISTER', 'UNDO'}
    def _get_pts(self, context, obj):
        pts = []
        for i in range(1, 4):
            v = context.scene.get(f"align_{obj}_p{i}")
            if v is None: return None
            pts.append(list(v))
        return pts
    def execute(self, context):
        P_pts = self._get_pts(context, "scan")
        Q_pts = self._get_pts(context, "foto")
        if not P_pts:
            self.report({'ERROR'}, "Grave os 3 pontos no Escaneamento!"); return {'CANCELLED'}
        if not Q_pts:
            self.report({'ERROR'}, "Grave os 3 pontos na Fotogrametria!"); return {'CANCELLED'}
        scan_obj = bpy.data.objects.get("ESCANEAMENTO") or context.active_object
        if not scan_obj:
            self.report({'ERROR'}, "Objeto ESCANEAMENTO não encontrado!"); return {'CANCELLED'}
        P = np.array(P_pts, dtype=float); Q = np.array(Q_pts, dtype=float)
        Pc = P.mean(0); Qc = Q.mean(0)
        H = (P - Pc).T @ (Q - Qc)
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        R = Vt.T @ np.diag([1., 1., d]) @ U.T
        t = Qc - R @ Pc
        R4 = Matrix([[R[0,0],R[0,1],R[0,2],t[0]],
                     [R[1,0],R[1,1],R[1,2],t[1]],
                     [R[2,0],R[2,1],R[2,2],t[2]],
                     [0,     0,     0,     1   ]])
        scan_obj.matrix_world = R4 @ scan_obj.matrix_world
        self.report({'INFO'}, "Escaneamento alinhado!"); return {'FINISHED'}


# ============================================================
# OPERADORES - PONTOS ANATÔMICOS
# ============================================================
class PLUSID_OT_LimparLandmarks(Operator):
    bl_idname = "plusid.limpar_landmarks"
    bl_label = "Limpar Todos os Landmarks"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        removidos = 0
        for code in LANDMARK_CODES:
            obj = bpy.data.objects.get(landmark_empty_name(code))
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)
                removidos += 1
        self.report({'INFO'}, f"{removidos} landmarks removidos.")
        return {'FINISHED'}


# ============================================================
# OPERADORES - ANÁLISE DE SIMETRIA
# ============================================================
class PLUSID_OT_AnalisarSimetria(Operator):
    bl_idname = "plusid.analisar_simetria"
    bl_label = "Analisar Simetria"
    bl_description = (
        "Usa os landmarks de linha média para definir o plano de simetria, "
        "espelha a malha e gera um heatmap de assimetria em mm"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecione a malha a analisar!")
            return {'CANCELLED'}

        # Pontos da linha média
        midline_pts = midline_landmarks()
        if len(midline_pts) < 2:
            self.report({'ERROR'},
                "Coloque pelo menos 2 landmarks de linha média (Glabela, Násio, etc.)!")
            return {'CANCELLED'}

        pts = np.array(midline_pts, dtype=float)

        # Calcula plano de simetria por mínimos quadrados
        centroid = pts.mean(axis=0)
        if len(pts) >= 3:
            _, _, Vt = np.linalg.svd(pts - centroid)
            normal = Vt[-1]  # menor variância = normal ao plano
        else:
            # Com 2 pontos, usa a diferença e o eixo Z como base
            diff = pts[1] - pts[0]
            diff = diff / (np.linalg.norm(diff) + 1e-9)
            up   = np.array([0, 0, 1.0])
            normal = np.cross(diff, up)
            normal = normal / (np.linalg.norm(normal) + 1e-9)

        # Obtém vértices do objeto em espaço de mundo
        mesh = obj.data
        mw   = np.array(obj.matrix_world)

        verts_local = np.array([v.co for v in mesh.vertices])
        verts_h     = np.hstack([verts_local, np.ones((len(verts_local), 1))])
        verts_world = (mw @ verts_h.T).T[:, :3]

        # Reflete vértices sobre o plano
        d            = np.dot(verts_world - centroid, normal)
        verts_mirror = verts_world - 2 * np.outer(d, normal)

        # Distância de cada vértice original ao mais próximo no espelho
        # (numpy broadcasting — pode ser lento para malhas grandes)
        n = len(verts_world)
        if n > 50000:
            self.report({'WARNING'},
                f"Malha com {n} vértices — a análise pode demorar alguns segundos.")

        # Processa em lotes para não explodir a memória
        batch = 1000
        dists = np.zeros(n)
        for i in range(0, n, batch):
            chunk = verts_world[i:i+batch]
            diff  = chunk[:, None, :] - verts_mirror[None, :, :]
            dists[i:i+batch] = np.sqrt((diff**2).sum(axis=2)).min(axis=1)

        # Converte para mm (Blender usa metros internamente)
        dists_mm = dists * 1000.0

        # Aplica heatmap como Color Attribute (Blender 4+)
        ca_name = "Assimetria_mm"
        if ca_name in mesh.color_attributes:
            mesh.color_attributes.remove(mesh.color_attributes[ca_name])
        col_attr = mesh.color_attributes.new(name=ca_name,
                                              type='FLOAT_COLOR', domain='POINT')

        max_d = np.percentile(dists_mm, 95)  # usa percentil 95 para evitar outliers
        max_d = max(max_d, 0.1)

        for i, d in enumerate(dists_mm):
            t = min(d / max_d, 1.0)
            # Verde (simétrico) → Amarelo → Vermelho (assimétrico)
            if t < 0.5:
                r, g = t * 2, 1.0
            else:
                r, g = 1.0, 1.0 - (t - 0.5) * 2
            col_attr.data[i].color = (r, g, 0.0, 1.0)

        # Ativa exibição de vertex colors
        obj.data.attributes.active_color = col_attr
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                with context.temp_override(area=area):
                    bpy.context.space_data.shading.color_type = 'VERTEX'
                break

        # Estatísticas
        media   = dists_mm.mean()
        maximo  = dists_mm.max()
        p95     = np.percentile(dists_mm, 95)

        # Salva resultado na cena
        context.scene["simetria_media_mm"]  = round(float(media),  2)
        context.scene["simetria_max_mm"]    = round(float(maximo), 2)
        context.scene["simetria_p95_mm"]    = round(float(p95),    2)

        self.report({'INFO'},
            f"Assimetria  média: {media:.2f} mm | "
            f"máx: {maximo:.2f} mm | P95: {p95:.2f} mm")
        return {'FINISHED'}


class PLUSID_OT_LimparHeatmap(Operator):
    bl_idname = "plusid.limpar_heatmap"
    bl_label = "Remover Heatmap"
    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'MESH':
            ca = obj.data.color_attributes.get("Assimetria_mm")
            if ca:
                obj.data.color_attributes.remove(ca)
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                with context.temp_override(area=area):
                    bpy.context.space_data.shading.color_type = 'MATERIAL'
                break
        # Limpa stats
        for k in ("simetria_media_mm", "simetria_max_mm", "simetria_p95_mm"):
            if k in context.scene:
                del context.scene[k]
        return {'FINISHED'}


# ============================================================
# OPERADORES - VERIFICAÇÃO PARA IMPRESSÃO
# ============================================================
class PLUSID_OT_VerificarImpressao(Operator):
    bl_idname = "plusid.verificar_impressao"
    bl_label = "Verificar"
    bl_description = "Analisa a malha e reporta problemas para impressão 3D"

    def execute(self, context):
        props = context.scene.plusid
        obj   = context.active_object

        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecione uma malha!")
            return {'CANCELLED'}

        mesh     = obj.data
        issues   = []
        warnings = []

        import bmesh as bm_module

        bm = bm_module.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # 1. Geometria não-manifold
        if props.chk_manifold:
            nm_edges = [e for e in bm.edges if not e.is_manifold]
            nm_verts = [v for v in bm.verts if not v.is_manifold]
            if nm_edges or nm_verts:
                issues.append(
                    f"Não-manifold: {len(nm_edges)} arestas, {len(nm_verts)} vértices")
            else:
                warnings.append("Manifold: OK")

        # 2. Normais invertidas (detecta faces com normal apontando para dentro da bbox)
        if props.chk_normais:
            centroid = sum((f.calc_center_median() for f in bm.faces), Vector()) / len(bm.faces)
            inverted = 0
            for f in bm.faces:
                to_center = centroid - f.calc_center_median()
                if f.normal.dot(to_center) > 0:
                    inverted += 1
            pct = inverted / max(len(bm.faces), 1) * 100
            if pct > 20:
                issues.append(f"Normais: {inverted} faces possivelmente invertidas ({pct:.0f}%)")
            else:
                warnings.append(f"Normais: OK ({inverted} suspeitas)")

        # 3. Espessura mínima (aproximação via edge length)
        if props.chk_espessura:
            min_mm = props.espessura_min
            # Usa escala do objeto para converter para mm
            scale  = (obj.scale[0] + obj.scale[1] + obj.scale[2]) / 3
            short  = [e for e in bm.edges
                      if e.calc_length() * scale * 1000 < min_mm]
            if short:
                warnings.append(
                    f"Espessura: {len(short)} arestas < {min_mm} mm — verifique paredes finas")
            else:
                warnings.append(f"Espessura: OK (> {min_mm} mm)")

        # 4. Volume (peça fechada)
        if props.chk_volume:
            if bm.is_valid:
                has_boundary = any(e for e in bm.edges if e.is_boundary)
                if has_boundary:
                    issues.append("Volume: malha aberta (tem bordas livres) — não imprime como sólido")
                else:
                    vol = sum(f.calc_area() for f in bm.faces)
                    warnings.append(f"Volume: malha fechada — OK")

        # 5. Dimensões da impressora
        if props.chk_dimensoes:
            bbox  = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            xs    = [v.x for v in bbox]; ys = [v.y for v in bbox]; zs = [v.z for v in bbox]
            dim_x = (max(xs) - min(xs)) * 1000
            dim_y = (max(ys) - min(ys)) * 1000
            dim_z = (max(zs) - min(zs)) * 1000
            px, py, pz = props.dim_impressora
            over = []
            if dim_x > px: over.append(f"X: {dim_x:.0f} > {px:.0f} mm")
            if dim_y > py: over.append(f"Y: {dim_y:.0f} > {py:.0f} mm")
            if dim_z > pz: over.append(f"Z: {dim_z:.0f} > {pz:.0f} mm")
            if over:
                issues.append(f"Dimensões excedem impressora: {', '.join(over)}")
            else:
                warnings.append(
                    f"Dimensões: OK ({dim_x:.0f} × {dim_y:.0f} × {dim_z:.0f} mm)")

        bm.free()

        # Salva resultado para exibir no painel
        context.scene["verif_issues"]   = issues
        context.scene["verif_warnings"] = warnings

        if issues:
            self.report({'WARNING'},
                f"{len(issues)} problema(s) encontrado(s). Veja o painel de Verificação.")
        else:
            self.report({'INFO'}, "Malha OK para impressão!")
        return {'FINISHED'}


class PLUSID_OT_CorrigirImpressao(Operator):
    bl_idname = "plusid.corrigir_impressao"
    bl_label = "Auto-corrigir"
    bl_description = "Aplica correções automáticas selecionadas"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.plusid
        obj   = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecione uma malha!")
            return {'CANCELLED'}

        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')

        if props.autofix_manifold:
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
            bpy.ops.mesh.fill_holes(sides=0)

        if props.autofix_normais:
            bpy.ops.mesh.normals_make_consistent(inside=False)

        bpy.ops.object.mode_set(mode='OBJECT')

        # Limpa resultados para forçar re-verificação
        for k in ("verif_issues", "verif_warnings"):
            if k in context.scene: del context.scene[k]

        self.report({'INFO'}, "Correções aplicadas. Execute a verificação novamente.")
        return {'FINISHED'}


# ============================================================
# OPERADORES - PREPARAÇÃO E EXPORTAÇÃO
# ============================================================
class PLUSID_OT_PrepararImpressao(Operator):
    bl_idname = "plusid.preparar_impressao"
    bl_label = "Preparar para Impressão 3D"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Selecione uma malha!"); return {'CANCELLED'}
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.mesh.fill_holes(sides=0)
        bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, "Malha preparada!"); return {'FINISHED'}

class PLUSID_OT_ExportarSTL(Operator):
    bl_idname = "plusid.exportar_stl"; bl_label = "Exportar STL"
    filepath: StringProperty(subtype='FILE_PATH')
    def invoke(self, context, event):
        props = context.scene.plusid
        nome  = f"{props.sobrenome_paciente}_{props.nome_paciente}_protese".strip("_")
        self.filepath = os.path.join(pasta_paciente(context), nome + ".stl")
        context.window_manager.fileselect_add(self); return {'RUNNING_MODAL'}
    def execute(self, context):
        bpy.ops.wm.stl_export(filepath=self.filepath,
                               export_selected_objects=True, ascii_format=False)
        self.report({'INFO'}, f"STL: {self.filepath}"); return {'FINISHED'}


# ============================================================
# OPERADORES - ATUALIZAÇÃO
# ============================================================
def _versao_local():
    vfile = os.path.join(os.path.dirname(__file__), "version.json")
    if os.path.exists(vfile):
        with open(vfile) as f:
            return tuple(json.load(f).get("version", [0, 0, 0]))
    return bl_info["version"]

def _versao_remota(url):
    with urllib.request.urlopen(url, timeout=8) as r:
        return tuple(json.loads(r.read()).get("version", [0, 0, 0]))

def _versao_str(v):
    return ".".join(str(x) for x in v)


class PLUSID_OT_SalvarURLs(Operator):
    bl_idname = "plusid.salvar_urls"
    bl_label = "Salvar URLs"
    bl_description = "Salva as URLs de atualização para uso futuro"

    def execute(self, context):
        props = context.scene.plusid
        _salvar_update_cfg(props.update_version_url, props.update_zip_url)
        self.report({'INFO'}, "URLs salvas!")
        return {'FINISHED'}


class PLUSID_OT_VerificarAtualizacao(Operator):
    bl_idname = "plusid.verificar_atualizacao"
    bl_label = "Verificar Atualização"
    bl_description = "Verifica se há uma versão mais nova disponível"

    def execute(self, context):
        props = context.scene.plusid
        cfg   = _carregar_update_cfg()
        url   = props.update_version_url or cfg.get("version_url", "")

        if not url:
            self.report({'WARNING'},
                "Configure a URL do version.json no painel de atualização!")
            return {'CANCELLED'}

        try:
            remota = _versao_remota(url)
            local  = _versao_local()
            context.scene["upd_versao_local"]  = _versao_str(local)
            context.scene["upd_versao_remota"] = _versao_str(remota)
            context.scene["upd_disponivel"]    = remota > local

            if remota > local:
                self.report({'WARNING'},
                    f"Nova versão disponível: {_versao_str(remota)} "
                    f"(atual: {_versao_str(local)})")
            else:
                self.report({'INFO'},
                    f"Plugin já está atualizado! (v{_versao_str(local)})")

        except Exception as e:
            self.report({'ERROR'}, f"Não foi possível verificar: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


class PLUSID_OT_AtualizarPlugin(Operator):
    bl_idname = "plusid.atualizar_plugin"
    bl_label = "Baixar e Instalar Atualização"
    bl_description = "Baixa a nova versão e instala automaticamente"

    def execute(self, context):
        props = context.scene.plusid
        cfg   = _carregar_update_cfg()
        url   = props.update_zip_url or cfg.get("zip_url", "")

        if not url:
            self.report({'WARNING'},
                "Configure a URL do ZIP no painel de atualização!")
            return {'CANCELLED'}

        try:
            # Baixa o zip para um temporário
            self.report({'INFO'}, "Baixando atualização…")
            tmpdir  = tempfile.mkdtemp()
            zippath = os.path.join(tmpdir, "mais_identidade.zip")

            urllib.request.urlretrieve(url, zippath)

            # Pasta de addons do Blender
            addons_dir = bpy.utils.user_resource('SCRIPTS', path="addons")
            destino    = os.path.join(addons_dir, "mais_identidade")

            # Backup da versão atual
            backup = destino + "_backup"
            if os.path.exists(backup):
                shutil.rmtree(backup)
            if os.path.exists(destino):
                shutil.copytree(destino, backup)

            # Extrai nova versão
            with zipfile.ZipFile(zippath, 'r') as z:
                z.extractall(addons_dir)

            shutil.rmtree(tmpdir, ignore_errors=True)

            # Recarrega o addon
            bpy.ops.preferences.addon_disable(module='mais_identidade')
            bpy.ops.preferences.addon_enable(module='mais_identidade')

            # Limpa flags
            for k in ("upd_versao_remota", "upd_disponivel"):
                if k in context.scene:
                    del context.scene[k]

            nova = _versao_local()
            self.report({'INFO'},
                f"Atualizado para v{_versao_str(nova)}! "
                f"Backup salvo em: {backup}")

        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Erro na atualização: {e}")
            return {'CANCELLED'}

        return {'FINISHED'}


# ============================================================
# PAINÉIS
# ============================================================
class PLUSID_PT_Paciente(Panel):
    bl_label = "Paciente"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"; bl_order = 0
    def draw(self, context):
        layout = self.layout; props = context.scene.plusid
        col = layout.column(align=True)
        col.prop(props, "nome_paciente"); col.prop(props, "sobrenome_paciente")
        pasta = context.scene.get("pasta_paciente", "")
        if pasta: layout.label(text=f"…/{os.path.basename(pasta)}", icon='FILE_FOLDER')
        layout.separator()
        layout.operator("plusid.salvar_paciente", icon='FILE_TICK')


class PLUSID_PT_Fotogrametria(Panel):
    bl_label = "1. Fotogrametria"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 1
    def draw(self, context):
        layout = self.layout; props = context.scene.plusid
        box = layout.box()
        box.label(text="Configurações:", icon='SETTINGS')
        box.prop(props, "motor_foto")
        box.prop(props, "qualidade_foto")
        if props.motor_foto == 'COLMAP':
            row = box.row()
            row.prop(props, "usar_gpu")
            if props.usar_gpu:
                row.label(text="Requer CUDA", icon='INFO')
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Pasta de fotos:")
        col.prop(props, "path_photo", text="")
        layout.separator()
        row = layout.row(); row.scale_y = 2.0
        row.operator("plusid.fotogrametria", icon='IMAGE_DATA')


class PLUSID_PT_Escala(Panel):
    bl_label = "2. Escala e Recorte"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 2
    def draw(self, context):
        layout = self.layout; props = context.scene.plusid
        box = layout.box()
        box.label(text="Escala pelos X marcados:", icon='DRIVER_DISTANCE')
        pt_a = context.scene.get("escala_ponto_a")
        pt_b = context.scene.get("escala_ponto_b")
        col = box.column(align=True)
        op_a = col.operator("plusid.snap_ponto",
            text="Ponto A  ✓" if pt_a else "Clicar no Ponto A",
            icon='CHECKMARK' if pt_a else 'CURSOR')
        op_a.modo = "escala_a"
        op_b = col.operator("plusid.snap_ponto",
            text="Ponto B  ✓" if pt_b else "Clicar no Ponto B",
            icon='CHECKMARK' if pt_b else 'CURSOR')
        op_b.modo = "escala_b"
        box.prop(props, "medida_real")
        row = box.row(); row.scale_y = 1.4; row.enabled = bool(pt_a and pt_b)
        row.operator("plusid.aplicar_escala", icon='DRIVER_DISTANCE')
        layout.separator()
        box2 = layout.box()
        box2.label(text="Recorte:", icon='FCURVE')
        col2 = box2.column(align=True)
        col2.operator("plusid.desenhar_linha", icon='LINE_DATA')
        col2.operator("plusid.cortar_desenho", icon='FCURVE')


class PLUSID_PT_Espelhamento(Panel):
    bl_label = "3. Espelhamento"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 3
    def draw(self, context):
        layout = self.layout; props = context.scene.plusid
        layout.prop(props, "eixo_espelho", expand=True)
        layout.separator()
        layout.operator("plusid.espelhar_protese", icon='MOD_MIRROR')


class PLUSID_PT_Escultura(Panel):
    bl_label = "4. Escultura"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 4
    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True); col.scale_y = 1.2
        col.operator("plusid.escultura_grab",   icon='HAND')
        col.operator("plusid.escultura_smooth", icon='MOD_SMOOTH')
        col.operator("plusid.escultura_clay",   icon='SCULPTMODE_HLT')
        col.operator("plusid.modo_objeto",      icon='OBJECT_DATA')
        layout.separator()
        box = layout.box()
        box.label(text="Limites:", icon='FCURVE')
        c = box.column(align=True)
        c.operator("plusid.desenhar_linha", text="Desenhar Limite", icon='LINE_DATA')
        c.operator("plusid.cortar_desenho", text="Cortar Limite",   icon='FCURVE')


class PLUSID_PT_Encaixe(Panel):
    bl_label = "5. Encaixe e Exportação"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 5
    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.operator("plusid.booleana_encaixe", icon='MOD_BOOLEAN')
        col.operator("plusid.booleana_uniao",   icon='MOD_CAST')
        layout.separator()
        layout.operator("plusid.preparar_impressao", icon='META_CUBE')
        layout.separator()
        layout.operator("plusid.exportar_stl", icon='EXPORT')


class PLUSID_PT_Biblioteca(Panel):
    bl_label = "6. Biblioteca de Próteses"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 6
    def draw(self, context):
        layout = self.layout; props = context.scene.plusid
        box = layout.box()
        box.label(text="Salvar peça:", icon='FILE_TICK')
        box.prop(props, "cat_biblioteca"); box.prop(props, "nome_biblioteca")
        box.operator("plusid.salvar_biblioteca", icon='EXPORT')
        layout.separator()
        box2 = layout.box()
        box2.label(text="Carregar peça:", icon='IMPORT')
        box2.prop(props, "cat_biblioteca")
        pecas = _listar_biblioteca(props.cat_biblioteca)
        if pecas:
            for nome in pecas:
                op = box2.operator("plusid.carregar_biblioteca", text=nome, icon='MESH_MONKEY')
                op.nome_peca = nome
        else:
            box2.label(text="Nenhuma peça nesta categoria.", icon='INFO')


class PLUSID_PT_ModeloTrabalho(Panel):
    bl_label = "7. Modelo de Trabalho"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 7
    def draw(self, context):
        layout = self.layout; props = context.scene.plusid
        layout.label(text="1. Desenhe o defeito:", icon='LINE_DATA')
        col = layout.column(align=True)
        col.operator("plusid.desenhar_linha", text="Desenhar Área", icon='LINE_DATA')
        col.operator("plusid.cortar_desenho", text="Isolar Área",   icon='FCURVE')
        layout.separator()
        box = layout.box()
        box.label(text="2. Configure o bloco:", icon='MOD_SOLIDIFY')
        box.prop(props, "prof_trabalho")
        box.prop(props, "oco_trabalho")
        if props.oco_trabalho: box.prop(props, "esp_parede")
        layout.separator()
        row = layout.row(); row.scale_y = 1.6
        row.operator("plusid.criar_modelo_trabalho", icon='MOD_SOLIDIFY')
        layout.separator()
        col2 = layout.column(align=True)
        col2.operator("plusid.preparar_impressao", icon='META_CUBE')
        col2.operator("plusid.exportar_stl",       icon='EXPORT')


class PLUSID_PT_Escaneamento(Panel):
    bl_label = "8. Escaneamento"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 8
    def _gravado(self, context, obj, slot):
        return context.scene.get(f"align_{obj}_p{slot}") is not None
    def draw(self, context):
        layout = self.layout
        layout.operator("plusid.importar_escaneamento", icon='FILE_3D')
        layout.separator()
        for label, chave in [("Pontos no Escaneamento:", "scan"),
                              ("Pontos na Fotogrametria:", "foto")]:
            box = layout.box(); box.label(text=label)
            for i in range(1, 4):
                gravado = self._gravado(context, chave, i)
                op = box.operator("plusid.snap_ponto",
                    text=f"P{i}  ✓" if gravado else f"Clicar P{i}",
                    icon='CHECKMARK' if gravado else 'CURSOR')
                op.modo = f"align_{chave}_{i}"
        layout.separator()
        todos = all(self._gravado(context, o, s)
                    for o in ("scan","foto") for s in range(1,4))
        row = layout.row(); row.scale_y = 1.6; row.enabled = todos
        row.operator("plusid.alinhar_escaneamento", icon='MOD_LATTICE')
        if not todos: layout.label(text="Grave todos os 6 pontos.", icon='INFO')


class PLUSID_PT_PontosAnatomicos(Panel):
    bl_label = "9. Pontos Anatômicos"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 9

    def draw(self, context):
        layout = self.layout

        layout.label(text="Clique no botão e depois na malha:", icon='INFO')
        layout.separator()

        # Linha média (amarelo)
        box_mid = layout.box()
        box_mid.label(text="Linha Média  (usados na simetria):", icon='SORTBYEXT')
        col = box_mid.column(align=True)
        for code, nome, desc, is_mid in LANDMARKS:
            if not is_mid:
                continue
            gravado = bpy.data.objects.get(landmark_empty_name(code)) is not None
            op = col.operator("plusid.snap_ponto",
                text=f"{nome}  ✓" if gravado else nome,
                icon='CHECKMARK' if gravado else 'CURSOR')
            op.modo = f"landmark_{code}"

        layout.separator()

        # Pares laterais (azul)
        box_lat = layout.box()
        box_lat.label(text="Pontos Laterais:", icon='SORTBYEXT')
        col2 = box_lat.column(align=True)
        for code, nome, desc, is_mid in LANDMARKS:
            if is_mid:
                continue
            gravado = bpy.data.objects.get(landmark_empty_name(code)) is not None
            op = col2.operator("plusid.snap_ponto",
                text=f"{nome}  ✓" if gravado else nome,
                icon='CHECKMARK' if gravado else 'CURSOR')
            op.modo = f"landmark_{code}"

        layout.separator()
        layout.operator("plusid.limpar_landmarks", icon='TRASH')


class PLUSID_PT_Simetria(Panel):
    bl_label = "10. Análise de Simetria"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 10

    def draw(self, context):
        layout = self.layout
        pts = midline_landmarks()
        n   = len(pts)

        if n < 2:
            layout.label(
                text=f"Coloque ≥ 2 landmarks de linha média ({n} colocados).",
                icon='ERROR')
        else:
            layout.label(text=f"{n} ponto(s) de linha média prontos.", icon='CHECKMARK')

        layout.separator()
        row = layout.row(); row.scale_y = 1.8; row.enabled = n >= 2
        row.operator("plusid.analisar_simetria", icon='MOD_WAVE')

        # Resultados
        media = context.scene.get("simetria_media_mm")
        maxi  = context.scene.get("simetria_max_mm")
        p95   = context.scene.get("simetria_p95_mm")

        if media is not None:
            layout.separator()
            box = layout.box()
            box.label(text="Resultado da última análise:", icon='DRIVER_DISTANCE')
            col = box.column(align=True)
            col.label(text=f"Média:   {media:.2f} mm")
            col.label(text=f"Máximo:  {maxi:.2f} mm")
            col.label(text=f"P95:     {p95:.2f} mm")
            box.separator()
            # Interpretação simples
            if media < 1.0:
                box.label(text="Excelente simetria!", icon='CHECKMARK')
            elif media < 2.5:
                box.label(text="Boa simetria.", icon='INFO')
            else:
                box.label(text="Assimetria significativa.", icon='ERROR')

        layout.separator()
        layout.operator("plusid.limpar_heatmap", icon='X')


class PLUSID_PT_Verificacao(Panel):
    bl_label = "11. Verificação para Impressão"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 11

    def draw(self, context):
        layout = self.layout
        props  = context.scene.plusid

        # Checklist
        box = layout.box()
        box.label(text="O que verificar:", icon='CHECKMARK')
        col = box.column(align=True)
        col.prop(props, "chk_manifold")
        col.prop(props, "chk_normais")
        col.prop(props, "chk_volume")
        row = col.row()
        row.prop(props, "chk_espessura")
        if props.chk_espessura:
            row.prop(props, "espessura_min", text="mm mín")
        col.prop(props, "chk_dimensoes")
        if props.chk_dimensoes:
            box.prop(props, "dim_impressora", text="X  Y  Z (mm)")

        layout.separator()
        layout.operator("plusid.verificar_impressao", icon='VIEWZOOM')

        # Resultados
        issues   = context.scene.get("verif_issues",   [])
        warnings = context.scene.get("verif_warnings", [])

        if issues or warnings:
            layout.separator()
            box2 = layout.box()

            for msg in issues:
                row = box2.row()
                row.alert = True
                row.label(text=msg, icon='ERROR')

            for msg in warnings:
                box2.label(text=msg, icon='CHECKMARK')

            if issues:
                layout.separator()
                box3 = layout.box()
                box3.label(text="Auto-correção:", icon='TOOL_SETTINGS')
                box3.prop(props, "autofix_manifold")
                box3.prop(props, "autofix_normais")
                box3.operator("plusid.corrigir_impressao", icon='SHADERFX')


class PLUSID_PT_Atualizacao(Panel):
    bl_label = "Atualização"; bl_region_type = 'UI'
    bl_space_type = 'VIEW_3D'; bl_category = "+ID"
    bl_options = {'DEFAULT_CLOSED'}; bl_order = 12

    def draw(self, context):
        layout = self.layout
        props  = context.scene.plusid
        cfg    = _carregar_update_cfg()

        # Versão atual
        local = _versao_local()
        box = layout.box()
        box.label(text=f"Versão instalada: {_versao_str(local)}", icon='INFO')

        remota     = context.scene.get("upd_versao_remota")
        disponivel = context.scene.get("upd_disponivel", False)

        if remota:
            if disponivel:
                row = box.row()
                row.alert = True
                row.label(text=f"Nova versão disponível: {remota}!", icon='ERROR')
            else:
                box.label(text="Plugin já está atualizado.", icon='CHECKMARK')

        layout.separator()

        # URLs
        box2 = layout.box()
        box2.label(text="URLs de atualização:", icon='URL')

        # Pré-preenche com valores salvos se as props estiverem vazias
        if not props.update_version_url and cfg.get("version_url"):
            props.update_version_url = cfg["version_url"]
        if not props.update_zip_url and cfg.get("zip_url"):
            props.update_zip_url = cfg["zip_url"]

        box2.prop(props, "update_version_url", text="version.json")
        box2.prop(props, "update_zip_url",     text="ZIP")
        box2.operator("plusid.salvar_urls", icon='FILE_TICK')

        layout.separator()

        col = layout.column(align=True)
        col.operator("plusid.verificar_atualizacao", icon='FILE_REFRESH')

        row = col.row()
        row.enabled = bool(disponivel)
        row.scale_y = 1.6
        row.operator("plusid.atualizar_plugin", icon='IMPORT')

        if not disponivel and remota:
            layout.label(text="Nenhuma atualização disponível.", icon='CHECKMARK')


# ============================================================
# REGISTRO
# ============================================================
classes = [
    PlusIDProperties,
    PLUSID_OT_SalvarPaciente,
    PLUSID_OT_Fotogrametria,
    PLUSID_OT_SnapPonto,
    PLUSID_OT_AplicarEscala,
    PLUSID_OT_DesenharLinha,
    PLUSID_OT_CortarDesenho,
    PLUSID_OT_EspelharProtese,
    PLUSID_OT_EsculturaGrab,
    PLUSID_OT_EsculturaSmooth,
    PLUSID_OT_EsculturaClay,
    PLUSID_OT_ModoObjeto,
    PLUSID_OT_BooleanaEncaixe,
    PLUSID_OT_BooleanaUniao,
    PLUSID_OT_SalvarBiblioteca,
    PLUSID_OT_CarregarBiblioteca,
    PLUSID_OT_CriarModeloTrabalho,
    PLUSID_OT_ImportarEscaneamento,
    PLUSID_OT_AlinharEscaneamento,
    PLUSID_OT_LimparLandmarks,
    PLUSID_OT_AnalisarSimetria,
    PLUSID_OT_LimparHeatmap,
    PLUSID_OT_VerificarImpressao,
    PLUSID_OT_CorrigirImpressao,
    PLUSID_OT_PrepararImpressao,
    PLUSID_OT_ExportarSTL,
    PLUSID_PT_Paciente,
    PLUSID_PT_Fotogrametria,
    PLUSID_PT_Escala,
    PLUSID_PT_Espelhamento,
    PLUSID_PT_Escultura,
    PLUSID_PT_Encaixe,
    PLUSID_PT_Biblioteca,
    PLUSID_PT_ModeloTrabalho,
    PLUSID_PT_Escaneamento,
    PLUSID_PT_PontosAnatomicos,
    PLUSID_PT_Simetria,
    PLUSID_PT_Verificacao,
    PLUSID_PT_Atualizacao,
    PLUSID_OT_SalvarURLs,
    PLUSID_OT_VerificarAtualizacao,
    PLUSID_OT_AtualizarPlugin,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.plusid = PointerProperty(type=PlusIDProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.plusid

if __name__ == "__main__":
    register()
