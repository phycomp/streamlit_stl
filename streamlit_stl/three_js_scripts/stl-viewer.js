class STLViewer extends HTMLElement {
  constructor() {
    super();
  }

  connectedCallback() {
    this.connected = true;

    const shadowRoot = this.attachShadow({ mode: 'open' });
    const container = document.createElement('div');
    container.style.width = '100%';
    container.style.height = '100%';

    shadowRoot.appendChild(container);

    if (!this.hasAttribute('model')) {
      throw new Error('model attribute is required');
    }

    const model = this.getAttribute('model');
    const color = parseInt(this.getAttribute('color').replace("#","0x"), 16);
    const auto_rotate = this.getAttribute('auto_rotate');
    const opacity = this.getAttribute('opacity');
    const shininess = Number(this.getAttribute('shininess'));
    let materialType = this.getAttribute('materialType');
    const cam_v_angle = Number(this.getAttribute('cam_v_angle'));
    const cam_h_angle = Number(this.getAttribute('cam_h_angle'));
    let cam_distance = Number(this.getAttribute('cam_distance'));
    const max_view_distance = Number(this.getAttribute('max_view_distance'));


    let camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 1, max_view_distance);
    let renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    window.addEventListener('resize', function () {
      renderer.setSize(container.clientWidth, container.clientHeight);
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
    }, false);
    let controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableZoom = true;
    let scene = new THREE.Scene();
    let hem_light = new THREE.HemisphereLight(0xffffff, 0x222222, 1.5);
    hem_light.position.set(0, 0, 1);
    scene.add(hem_light);


    let dirLight = new THREE.DirectionalLight(0xffffff,1.5);
    dirLight.position.set(-1, 1, 0);
    scene.add(dirLight);
    

    new THREE.STLLoader().load(model, (geometry) => {
      let material = new THREE.MeshPhongMaterial({
        color: color,
        shininess: shininess,
        opacity: opacity, 
        transparent: true,
      });

      let flat = new THREE.MeshBasicMaterial({
        color: color,
        opacity: opacity, 
        transparent: true,
      });

      let wireframe = new THREE.MeshBasicMaterial({
        color: color,
        wireframe: true, 
        wireframeLinewidth: 40 
      });

      if (materialType == 'material') {
        materialType = material;
      } else if (materialType == 'flat') {
        materialType = flat;
      }
      else {
        materialType = wireframe;
      }
        
      let mesh = new THREE.Mesh(geometry, materialType);
      //mesh = THREE.SceneUtils.createMultiMaterialObject(geometry, [material, lines]);

      // avoid singularities in the rotation matrix with vertical angles of 0 and 180 degrees
      if (cam_v_angle % 180 == 0) {
        mesh.rotation.z = cam_h_angle * (Math.PI / 180);
      }
      scene.add(mesh);

      let middle = new THREE.Vector3();
      geometry.computeBoundingBox();
      geometry.boundingBox.getCenter(middle);
      mesh.geometry.applyMatrix4(new THREE.Matrix4().makeTranslation(-middle.x, -middle.y, -middle.z));
      let largestDimension = Math.max(geometry.boundingBox.max.x, geometry.boundingBox.max.y, geometry.boundingBox.max.z)
      if (cam_distance == 0) {
        cam_distance = largestDimension * 3;
      }

      // Convert degrees to radians
      const phi = cam_v_angle * (Math.PI / 180);
      const theta = cam_h_angle * (Math.PI / 180);
      camera.position.x = cam_distance * Math.sin(phi) * Math.cos(theta);
      camera.position.y = cam_distance * Math.sin(phi) * Math.sin(theta);
      camera.position.z = cam_distance * Math.cos(phi);
      camera.up.set( 0, 0, 1 );
      camera.lookAt(new THREE.Vector3(0,0,0)); 

      if (auto_rotate == 'true') {
        controls.autoRotate = true;
        controls.autoRotateSpeed = .5;
      }
      let animate = () => {
        controls.update();
        renderer.render(scene, camera);
        if (this.connected) {
          requestAnimationFrame(animate);
        }
      };
      animate();
    });
  }

  disconnectedCallback() {
    this.connected = false;
  }
}

customElements.define('stl-viewer', STLViewer);